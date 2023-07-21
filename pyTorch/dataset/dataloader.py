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
import pdb


class utils:
    def __init__(self, num_ensemble, dim_x, dim_z):
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z

    def multivariate_normal_sampler(self, mean, cov, k):
        sampler = MultivariateNormal(mean, cov)
        return sampler.sample((k,))

    def format_state(self, state):
        state = repeat(state, "k dim -> n k dim", n=self.num_ensemble)
        state = rearrange(state, "n k dim -> (n k) dim")
        cov = torch.eye(self.dim_x) * 0.05
        init_dist = self.multivariate_normal_sampler(
            torch.zeros(self.dim_x), cov, self.num_ensemble
        )
        state = state + init_dist
        state = state.to(dtype=torch.float32)
        return state


class CarDataset(Dataset):
    # Basic Instantiation
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        if self.mode == "train":
            self.dataset_path = self.args.train.data_path
            self.num_ensemble = self.args.train.num_ensemble
        elif self.mode == "test":
            self.dataset_path = self.args.test.data_path
            self.num_ensemble = self.args.test.num_ensemble
        self.dataset = pickle.load(open(self.dataset_path, "rb"))
        self.dataset_length = len(self.dataset)
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.utils_ = utils(
            self.num_ensemble, self.args.train.dim_x, self.args.train.dim_z
        )

    def process_image(self, img_path):
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0
        return img_array

    # Length of the Dataset
    def __len__(self):
        return self.dataset_length - 2

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        # make sure always take the data from the same sequence
        not_valid = True
        while not_valid:
            try:
                if self.dataset[idx][0] == self.dataset[idx + 1][0]:
                    not_valid = False
                else:
                    idx = random.randint(0, self.dataset_length)
            except:
                idx = random.randint(0, self.dataset_length)

        # the observation to the model
        pre = self.dataset[idx][2]
        gt = self.dataset[idx + 1][2]
        pre = torch.tensor(pre, dtype=torch.float32)
        pre = rearrange(pre, "(k dim) -> k dim", k=1)
        ensemble_pre = self.utils_.format_state(pre)

        gt = torch.tensor(gt, dtype=torch.float32)
        gt = rearrange(gt, "(k dim) -> k dim", k=1)

        # gt image
        img_path = "./dataset" + self.dataset[idx + 1][3]
        gt_image = self.process_image(img_path)
        gt_image = torch.tensor(gt_image, dtype=torch.float32)
        gt_image = rearrange(gt_image, "h w ch -> ch h w")

        data = (pre, ensemble_pre, gt, gt_image)

        return data
