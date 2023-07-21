import os
import random
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from einops import rearrange, repeat
from torch.distributions.multivariate_normal import MultivariateNormal
import math


class UR5_sim_dataloader(Dataset):
    # Basic Instantiation
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        if self.mode == "train":
            self.dataset_path = self.args.train.data_path
            self.parent_path = "/tf/datasets/"
            self.num_ensemble = self.args.train.num_ensemble
        elif self.mode == "test":
            self.dataset_path = self.args.test.data_path
            self.parent_path = "/tf/datasets/"
            self.num_ensemble = self.args.test.num_ensemble
        self.dataset = pickle.load(open(self.dataset_path, "rb"))
        self.dataset_length = len(self.dataset)
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.dim_a = self.args.train.dim_a
        self.win_size = self.args.train.win_size

    def add_sp_noise(self, img, mode):
        # Getting the dimensions of the image
        if mode == "rgb":
            row, col, _ = img.shape
        else:
            row, col = img.shape
        # Randomly pick some pixels in the
        # image for coloring them white
        # Pick a random number between 300 and 10000
        scale = 0.005
        number_of_pixels = int(224 * 224 * scale)
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord = random.randint(0, row - 1)
            # Pick a random x coordinate
            x_coord = random.randint(0, col - 1)
            if mode == "rgb":
                # Color that pixel to white
                img[y_coord][x_coord] = 1
            else:
                img[y_coord][x_coord] = 10
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord = random.randint(0, row - 1)
            # Pick a random x coordinate
            x_coord = random.randint(0, col - 1)
            # Color that pixel to black
            img[y_coord][x_coord] = 0
        return img

    def add_blur(self, img):
        # ksize
        ksize = (8, 8)
        img = cv2.blur(img, ksize)
        # img = cv2.GaussianBlur(img, ksize, 0)
        return img

    def process_image(self, img_path):
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0
        # img_array = self.add_sp_noise(img_array, "rgb")
        img_array = self.add_blur(img_array)
        return img_array

    def process_depth(self, img_path):
        img_array = np.load(img_path)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = np.float32(img_array) / 1000
        img_array[img_array > 30] = 0
        img_array = self.add_blur(img_array)
        # img_array = self.add_sp_noise(img_array, "depth")
        return img_array

    def get_data(self, idx):
        # the gt input to the model
        gt_input = []
        for i in range(self.win_size):
            joint = self.dataset[idx + i][2]
            EE = self.dataset[idx + i][4]
            tmp = np.concatenate((joint, EE), axis=None)
            gt_input.append(tmp)
        gt_input = np.array(gt_input)
        gt_input = gt_input + np.random.normal(0, 0.1, gt_input.shape)
        gt_input = torch.tensor(gt_input, dtype=torch.float32)

        # get the gt
        gt_joint = self.dataset[idx + self.win_size][2]
        gt_EE = self.dataset[idx + self.win_size][4]
        gt = np.concatenate((gt_joint, gt_EE), axis=None)

        # obs rbg img
        img_path = self.parent_path + self.dataset[idx + self.win_size][5]
        rgb = self.process_image(img_path)

        # obs depth img
        img_path = self.parent_path + self.dataset[idx + self.win_size][6]
        depth = self.process_depth(img_path)

        # obs joint angle
        obs_joint = np.array(self.dataset[idx + self.win_size][2])
        obs_joint = obs_joint + np.random.normal(0, 0.1, obs_joint.shape)

        # convert to tensor
        obs_joint = torch.tensor(obs_joint, dtype=torch.float32)
        obs_joint = rearrange(obs_joint, "(k dim) -> k dim", k=1)

        gt = torch.tensor(gt, dtype=torch.float32)
        gt = rearrange(gt, "(k dim) -> k dim", k=1)

        # images
        rgb = torch.tensor(rgb, dtype=torch.float32)
        rgb = rearrange(rgb, "h w ch -> ch h w")

        depth = torch.tensor(depth, dtype=torch.float32)
        depth = rearrange(depth, "(ch h) w -> ch h w", ch=1)

        out = (gt_input, rgb, depth, obs_joint, gt)
        return out

    # Length of the Dataset
    def __len__(self):
        # self.dataset_length = 50
        return self.dataset_length - self.win_size - 3

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        # make sure always take the data from the same sequence
        not_valid = True
        while not_valid:
            try:
                if self.dataset[idx][0] == self.dataset[idx + self.win_size + 4][
                    0
                ] and str(self.dataset[idx][0]) != str(1507):
                    not_valid = False
                else:
                    idx = random.randint(0, self.dataset_length)
            except:
                idx = random.randint(0, self.dataset_length)

        # data from t-1
        out_1 = self.get_data(idx)

        # data from t
        idx = idx + 1
        out_2 = self.get_data(idx)

        return out_1, out_2


class UR5_real_dataloader(Dataset):
    # Basic Instantiation
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        if self.mode == "train":
            self.dataset_path = self.args.train.data_path
            self.parent_path = "/tf/datasets/"
            self.num_ensemble = self.args.train.num_ensemble
        elif self.mode == "test":
            self.dataset_path = self.args.test.data_path
            self.parent_path = "/tf/datasets/"
            self.num_ensemble = self.args.test.num_ensemble
        self.dataset = pickle.load(open(self.dataset_path, "rb"))
        self.dataset_length = len(self.dataset)
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.dim_a = self.args.train.dim_a
        self.win_size = self.args.train.win_size

    def process_image(self, img_path):
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0
        return img_array

    # Length of the Dataset
    def __len__(self):
        return self.dataset_length - self.win_size - 3

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        # make sure always take the data from the same sequence
        not_valid = True
        while not_valid:
            try:
                if self.dataset[idx][0] == self.dataset[idx + self.win_size + 2][0]:
                    not_valid = False
                else:
                    idx = random.randint(0, self.dataset_length)
            except:
                idx = random.randint(0, self.dataset_length)

        #################### all data from t-1 ####################
        # the gt input to the model
        gt_input = []
        for i in range(self.win_size):
            joint = self.dataset[idx + i][2]
            EE = self.dataset[idx + i][4]
            tmp = np.concatenate((joint, EE), axis=None)
            gt_input.append(tmp)
        gt_input = np.array(gt_input)
        # gt_input = gt_input + np.random.normal(0, 0.1, gt_input.shape)
        gt_input = torch.tensor(gt_input, dtype=torch.float32)

        # get the gt
        gt_joint = self.dataset[idx + self.win_size][2]
        gt_EE = self.dataset[idx + self.win_size][4]
        gt = np.concatenate((gt_joint, gt_EE), axis=None)

        # obs rbg img
        img_path = self.parent_path + self.dataset[idx + self.win_size][5]
        rgb = self.process_image(img_path)

        # obs joint angle
        obs_joint = np.array(self.dataset[idx + self.win_size][2])
        obs_joint = obs_joint + np.random.normal(0, 0.1, obs_joint.shape)

        # convert to tensor
        obs_joint = torch.tensor(obs_joint, dtype=torch.float32)
        obs_joint = rearrange(obs_joint, "(k dim) -> k dim", k=1)

        gt = torch.tensor(gt, dtype=torch.float32)
        gt = rearrange(gt, "(k dim) -> k dim", k=1)

        # images
        rgb = torch.tensor(rgb, dtype=torch.float32)
        rgb = rearrange(rgb, "h w ch -> ch h w")

        out_1 = (gt_input, rgb, obs_joint, gt)

        #################### all data from t ####################
        idx = idx + 1

        # the gt input to the model
        gt_input = []
        for i in range(self.win_size):
            joint = self.dataset[idx + i][2]
            EE = self.dataset[idx + i][4]
            tmp = np.concatenate((joint, EE), axis=None)
            gt_input.append(tmp)
        gt_input = np.array(gt_input)
        gt_input = torch.tensor(gt_input, dtype=torch.float32)

        # get the gt
        gt_joint = self.dataset[idx + self.win_size][2]
        gt_EE = self.dataset[idx + self.win_size][4]
        gt = np.concatenate((gt_joint, gt_EE), axis=None)

        # obs rbg img
        img_path = self.parent_path + self.dataset[idx + self.win_size][5]
        rgb = self.process_image(img_path)

        # obs joint angle
        obs_joint = np.array(self.dataset[idx + self.win_size][2])
        obs_joint = obs_joint + np.random.normal(0, 0.1, obs_joint.shape)

        # convert to tensor
        obs_joint = torch.tensor(obs_joint, dtype=torch.float32)
        obs_joint = rearrange(obs_joint, "(k dim) -> k dim", k=1)

        gt = torch.tensor(gt, dtype=torch.float32)
        gt = rearrange(gt, "(k dim) -> k dim", k=1)

        # images
        rgb = torch.tensor(rgb, dtype=torch.float32)
        rgb = rearrange(rgb, "h w ch -> ch h w")

        out_2 = (gt_input, rgb, obs_joint, gt)

        return out_1, out_2


class UR5_push_dataloader(Dataset):
    # Basic Instantiation
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        if self.mode == "train":
            self.dataset_path = self.args.train.data_path
            self.parent_path = "/tf/datasets/"
            self.num_ensemble = self.args.train.num_ensemble
        elif self.mode == "test":
            self.dataset_path = self.args.test.data_path
            self.parent_path = "/tf/datasets/"
            self.num_ensemble = self.args.test.num_ensemble
        self.dataset = pickle.load(open(self.dataset_path, "rb"))
        self.dataset_length = len(self.dataset)
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.dim_a = self.args.train.dim_a
        self.win_size = self.args.train.win_size

    def process_image(self, img_path):
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0
        return img_array

    def process_depth(self, img_path):
        img_array = np.load(img_path)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = np.float32(img_array) / 1000
        img_array[img_array > 30] = 0
        return img_array

    def get_data(self, idx):
        # the gt input to the model
        gt_input = []
        for i in range(self.win_size):
            joint = self.dataset[idx + i][2]
            EE = self.dataset[idx + i][4]
            obj = self.dataset[idx + i][-2]

            tmp = np.concatenate((joint, EE, obj), axis=None)
            gt_input.append(tmp)
        gt_input = np.array(gt_input)
        gt_input = gt_input + np.random.normal(0, 0.1, gt_input.shape)
        gt_input = torch.tensor(gt_input, dtype=torch.float32)

        # get the gt
        gt_joint = self.dataset[idx + self.win_size][2]
        gt_EE = self.dataset[idx + self.win_size][4]
        gt_obj = self.dataset[idx + self.win_size][-2]
        gt = np.concatenate((gt_joint, gt_EE, gt_obj), axis=None)

        # obs rbg img
        img_path = self.parent_path + self.dataset[idx + self.win_size][5]
        rgb = self.process_image(img_path)

        # obs depth img
        img_path = self.parent_path + self.dataset[idx + self.win_size][6]
        depth = self.process_depth(img_path)

        # obs joint angle
        obs_joint = np.array(self.dataset[idx + self.win_size][2])
        obs_joint = obs_joint + np.random.normal(0, 0.1, obs_joint.shape)

        # F/T sensor
        obs_force = np.array(self.dataset[idx + self.win_size][-1])

        # convert to tensor
        obs_joint = torch.tensor(obs_joint, dtype=torch.float32)
        obs_joint = rearrange(obs_joint, "(k dim) -> k dim", k=1)

        obs_force = torch.tensor(obs_force, dtype=torch.float32)
        obs_force = rearrange(obs_force, "(k dim) -> k dim", k=1)

        gt = torch.tensor(gt, dtype=torch.float32)
        gt = rearrange(gt, "(k dim) -> k dim", k=1)

        # images
        rgb = torch.tensor(rgb, dtype=torch.float32)
        rgb = rearrange(rgb, "h w ch -> ch h w")

        depth = torch.tensor(depth, dtype=torch.float32)
        depth = rearrange(depth, "(ch h) w -> ch h w", ch=1)

        out = (gt_input, rgb, depth, obs_joint, obs_force, gt)
        return out

    # Length of the Dataset
    def __len__(self):
        # self.dataset_length = 50
        return self.dataset_length - self.win_size - 3

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        # make sure always take the data from the same sequence
        not_valid = True
        while not_valid:
            try:
                if self.dataset[idx][0] == self.dataset[idx + self.win_size + 4][
                    0
                ] and str(self.dataset[idx][0]) != str(1507):
                    not_valid = False
                else:
                    idx = random.randint(0, self.dataset_length)
            except:
                idx = random.randint(0, self.dataset_length)

        # data from t-1
        out_1 = self.get_data(idx)

        # data from t
        idx = idx + 1
        out_2 = self.get_data(idx)

        return out_1, out_2
