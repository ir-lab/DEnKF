import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import CarDataset
from model import Ensemble_KF_low
from model import DEnKF
from optimizer import build_optimizer
from optimizer import build_lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import random
import pickle


class Engine:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.train.batch_size
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.dim_a = self.args.train.dim_a
        self.num_ensemble = self.args.train.num_ensemble
        self.global_step = 0
        self.mode = self.args.mode.mode
        self.dataset = CarDataset(self.args, self.mode)
        self.model = DEnKF(self.num_ensemble, self.dim_x, self.dim_z)
        # Check model type
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.cuda()

    def test(self):
        test_dataset = CarDataset(self.args, "test")
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        step = 0
        data_out = {}
        data_save = []
        ensemble_save = []
        gt_save = []
        obs_save = []
        for data in test_dataloader:
            data = [item.to(self.device) for item in data]
            state_ensemble = data[1]
            state_pre = data[0]
            obs = data[3]
            state_gt = data[2]

            with torch.no_grad():
                if step == 0:
                    ensemble = state_ensemble
                    state = state_pre
                else:
                    ensemble = ensemble
                    state = state
                input_state = (ensemble, state)
                obs_action = obs
                output = self.model(obs_action, input_state)

                ensemble = output[0]  # -> ensemble estimation
                state = output[1]  # -> final estimation
                obs_p = output[3]  # -> learned observation

                final_ensemble = ensemble  # -> make sure these variables are tensor
                final_est = state
                obs_est = obs_p

                final_ensemble = final_ensemble.cpu().detach().numpy()
                final_est = final_est.cpu().detach().numpy()
                obs_est = obs_est.cpu().detach().numpy()
                state_gt = state_gt.cpu().detach().numpy()

                data_save.append(final_est)
                ensemble_save.append(final_ensemble)
                gt_save.append(state_gt)
                obs_save.append(obs_est)
                step = step + 1

        data_out["state"] = data_save
        data_out["ensemble"] = ensemble_save
        data_out["gt"] = gt_save
        data_out["observation"] = obs_save

        save_path = os.path.join(
            self.args.train.eval_summary_directory,
            self.args.train.model_name,
            "eval-result-{}.pkl".format(self.global_step),
        )

        with open(save_path, "wb") as f:
            pickle.dump(data_out, f)

    def train(self):
        # # Load the pretrained model
        # if torch.cuda.is_available():
        #     checkpoint = torch.load(self.args.test.checkpoint_path)
        #     self.model.load_state_dict(checkpoint["model"])
        # else:
        #     checkpoint = torch.load(
        #         self.args.test.checkpoint_path, map_location=torch.device("cpu")
        #     )
        #     self.model.load_state_dict(checkpoint["model"])

        mse_criterion = nn.MSELoss()
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )
        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Total number of parameters: ", pytorch_total_params)

        # Create optimizer
        optimizer_ = build_optimizer(
            [self.model],
            self.args.network.name,
            self.args.optim.optim,
            self.args.train.learning_rate,
            self.args.train.weight_decay,
            self.args.train.adam_eps,
        )

        # Create LR scheduler
        if self.args.mode.mode == "train":
            num_total_steps = self.args.train.num_epochs * len(dataloader)
            scheduler = build_lr_scheduler(
                optimizer_,
                self.args.optim.lr_scheduler,
                self.args.train.learning_rate,
                num_total_steps,
                self.args.train.end_learning_rate,
            )
        # Epoch calculations
        steps_per_epoch = len(dataloader)
        num_total_steps = self.args.train.num_epochs * steps_per_epoch
        epoch = self.global_step // steps_per_epoch
        duration = 0

        # tensorboard writer
        self.writer = SummaryWriter(
            f"./experiments/{self.args.train.model_name}/summaries"
        )

        ####################################################################################################
        # MAIN TRAINING LOOP
        ####################################################################################################

        while epoch < self.args.train.num_epochs:
            step = 0
            for data in dataloader:
                data = [item.to(self.device) for item in data]
                state_ensemble = data[1]
                state_pre = data[0]
                obs = data[3]
                state_gt = data[2]

                # define the training curriculum
                optimizer_.zero_grad()
                before_op_time = time.time()

                # forward pass
                input_state = (state_ensemble, state_pre)
                obs_action = obs
                output = self.model(obs_action, input_state)

                final_est = output[1]  # -> final estimation
                inter_est = output[2]  # -> state transition output
                obs_est = output[3]  # -> learned observation
                hx = output[5]  # -> observation output

                # calculate loss
                loss_1 = mse_criterion(final_est, state_gt)
                loss_2 = mse_criterion(inter_est, state_gt)
                loss_3 = mse_criterion(obs_est, state_gt)
                loss_4 = mse_criterion(hx, state_gt)

                if epoch <= 10:
                    final_loss = loss_3
                else:
                    final_loss = loss_1 + loss_2 + loss_3 + loss_4

                # final_loss = loss_1 + loss_2 + loss_3 + loss_4

                # back prop
                final_loss.backward()
                optimizer_.step()
                current_lr = optimizer_.param_groups[0]["lr"]

                # verbose
                if self.global_step % self.args.train.log_freq == 0:
                    string = "[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}"
                    self.logger.info(
                        string.format(
                            epoch,
                            step,
                            steps_per_epoch,
                            self.global_step,
                            current_lr,
                            final_loss,
                        )
                    )
                    if np.isnan(final_loss.cpu().item()):
                        self.logger.warning("NaN in loss occurred. Aborting training.")
                        return -1

                # tensorboard
                duration += time.time() - before_op_time
                if (
                    self.global_step
                    and self.global_step % self.args.train.log_freq == 0
                ):
                    self.writer.add_scalar(
                        "end_to_end_loss", final_loss.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "transition model", loss_2.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "sensor_model", loss_3.cpu().item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "observation_model", loss_4.cpu().item(), self.global_step
                    )
                    # self.writer.add_scalar('learning_rate', current_lr, self.global_step)

                step += 1
                self.global_step += 1
                if scheduler is not None:
                    scheduler.step(self.global_step)

            # Save a model based of a chosen save frequency
            if self.global_step != 0 and (epoch + 1) % self.args.train.save_freq == 0:
                checkpoint = {
                    "global_step": self.global_step,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer_.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.args.train.log_directory,
                        self.args.train.model_name,
                        "model-{}".format(self.global_step),
                    ),
                )

            # online evaluation
            if (
                self.args.mode.do_online_eval
                and self.global_step != 0
                and epoch + 1 >= 10
                and (epoch + 1) % self.args.train.eval_freq == 0
            ):
                time.sleep(0.1)
                self.model.eval()
                self.test()
                self.model.train()

            # Update epoch
            epoch += 1

    def online_test(self):
        # Load the pretrained model
        if torch.cuda.is_available():
            checkpoint = torch.load(self.args.test.checkpoint_path)
            self.model.load_state_dict(checkpoint["model"])
        else:
            checkpoint = torch.load(
                self.args.test.checkpoint_path, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        test_dataset = CarDataset(self.args, "test")
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        step = 0
        data_out = {}
        data_save = []
        ensemble_save = []
        gt_save = []
        obs_save = []
        for data in test_dataloader:
            data = [item.to(self.device) for item in data]
            state_ensemble = data[1]
            state_pre = data[0]
            obs = data[3]
            state_gt = data[2]

            with torch.no_grad():
                if step == 0:
                    ensemble = state_ensemble
                    state = state_pre
                else:
                    ensemble = ensemble
                    state = state
                input_state = (ensemble, state)
                obs_action = obs
                output = self.model(obs_action, input_state)

                ensemble = output[0]  # -> ensemble estimation
                state = output[1]  # -> final estimation
                obs_p = output[3]  # -> learned observation

                final_ensemble = ensemble  # -> make sure these variables are tensor
                final_est = state
                obs_est = obs_p

                final_ensemble = final_ensemble.cpu().detach().numpy()
                final_est = final_est.cpu().detach().numpy()
                obs_est = obs_est.cpu().detach().numpy()
                state_gt = state_gt.cpu().detach().numpy()

                data_save.append(final_est)
                ensemble_save.append(final_ensemble)
                gt_save.append(state_gt)
                obs_save.append(obs_est)
                step = step + 1

        data_out["state"] = data_save
        data_out["ensemble"] = ensemble_save
        data_out["gt"] = gt_save
        data_out["observation"] = obs_save

        save_path = os.path.join(
            self.args.train.eval_summary_directory,
            self.args.train.model_name,
            "eval-result-{}.pkl".format(self.global_step),
        )

        with open(save_path, "wb") as f:
            pickle.dump(data_out, f)
