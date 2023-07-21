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


class KITTI_dataloader(Dataset):
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

    def process_image(self, img_path_1, img_path_2):
        img_2 = cv2.imread(img_path_2)
        img_1 = cv2.imread(img_path_1)
        img_2 = cv2.resize(img_2, (150, 50), interpolation=cv2.INTER_LINEAR)
        img_1 = cv2.resize(img_1, (150, 50), interpolation=cv2.INTER_LINEAR)
        img_2_ = img_2.astype(np.float32) / 255.0
        img_1_ = img_1.astype(np.float32) / 255.0
        ###########
        diff = img_2_ - img_1_
        diff = diff * 0.5 + 0.5
        ###########
        img = np.concatenate((img_2_, diff), axis=-1)
        return img

    def get_data(self, idx, offset):
        # the gt input to the model
        gt_input = []
        for i in range(self.win_size):
            tmp = np.array(self.dataset[idx + i][1][-2:])
            # tmp[:2] = tmp[:2] + offset
            gt_input.append(tmp)
        gt_input = np.array(gt_input)
        gt_input = gt_input + np.random.normal(0, 0.1, gt_input.shape)
        gt_input = torch.tensor(gt_input, dtype=torch.float32)

        # get the gt
        gt = np.array(self.dataset[idx + self.win_size][1][-2:])
        # gt[:2] = gt[:2] + offset

        # obs rbg img
        img_path_1 = self.parent_path + self.dataset[idx + self.win_size][3][0]
        img_path_2 = self.parent_path + self.dataset[idx + self.win_size][3][1]
        rgb = self.process_image(img_path_1, img_path_2)

        gt = torch.tensor(gt, dtype=torch.float32)
        gt = rearrange(gt, "(k dim) -> k dim", k=1)

        # images
        rgb = torch.tensor(rgb, dtype=torch.float32)
        rgb = rearrange(rgb, "h w ch -> ch h w")

        out = (gt_input, rgb, gt)
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
                if self.dataset[idx][-1] == self.dataset[idx + self.win_size + 4][-1]:
                    not_valid = False
                else:
                    idx = random.randint(0, self.dataset_length)
            except:
                idx = random.randint(0, self.dataset_length)

        # offsets
        if self.mode == "train":
            # offset = random.uniform(0, 2)
            offset = 0
        else:
            offset = 0

        # data from t-1
        out_1 = self.get_data(idx, offset)

        # data from t
        idx = idx + 1
        out_2 = self.get_data(idx, offset)

        return out_1, out_2
