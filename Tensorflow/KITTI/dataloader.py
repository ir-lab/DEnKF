import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import random
import tensorflow as tf
import time
import pickle
import pdb
import tensorflow_probability as tfp
import csv
import cv2
import re

class transform:
    def __init__(self):
        super(transform, self).__init__()
        parameters = pickle.load(open('parameters.pkl', 'rb'))
        self.v_m = parameters['v_m']
        self.v_std = parameters['v_std']
        self.theta_dot_m = parameters['theta_dot_m']
        self.theta_dot_std = parameters['theta_dot_std']

    def for_transform(self, state):
        '''
        state -> [batch_size, num_ensemble, dim_x]
        '''
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_x = state.shape[2]

        state = tf.reshape(state, [batch_size * num_ensemble, dim_x])
        v = state[:,0]
        theta_dot = state[:,1]

        v = (v - self.v_m)/self.v_std
        theta_dot = (theta_dot - self.theta_dot_m)/self.theta_dot_std

        state = tf.stack([v, theta_dot])
        state = tf.transpose(state)
        state = tf.reshape(state, [batch_size, num_ensemble, dim_x])
        return state

    def inv_transform(self, state):
        '''
        state -> [batch_size, num_ensemble, dim_x]
        '''
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_x = state.shape[2]

        state = tf.reshape(state, [batch_size * num_ensemble, dim_x])
        v = state[:,0]
        theta_dot = state[:,1]

        v = v * self.v_std + self.v_m
        theta_dot = theta_dot * self.theta_dot_std + self.theta_dot_m

        state = tf.stack([v, theta_dot])
        state = tf.transpose(state)
        state = tf.reshape(state, [batch_size, num_ensemble, dim_x])
        return state

class DataLoader:
    def __init__(self):
        # self.dataset_path = '/Users/xiao.lu/project/KITTI_dataset/' 
        self.dataset_path = 'dataset/' 
        self.transform_ = transform()

    def img_augmentation(self, img):
        value = 30
        value = int(random.uniform(-value, value))
        img = img + value
        img[:,:,:][img[:,:,:]>255]  = 255
        img[:,:,:][img[:,:,:]<0]  = 0
        img = img.astype(np.uint8)

        value = random.uniform(0.5, 2)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value 
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def preprocessing(self, data, train_mode):
        # img_2 = cv2.imread(self.dataset_path+data[3][1])
        # img_1 = cv2.imread(self.dataset_path+data[3][0])
        img_2 = cv2.imread(data[3][1])
        img_1 = cv2.imread(data[3][0])
        if train_mode == True:
            img_2 = self.img_augmentation(img_2)
            img_1 = self.img_augmentation(img_1)
        # 150, 50
        # 640, 192
        img_2 = cv2.resize(img_2, (640, 192), interpolation=cv2.INTER_LINEAR)
        img_1 = cv2.resize(img_1, (640, 192), interpolation=cv2.INTER_LINEAR)
        img_2_ = img_2.astype(np.float32)/255.
        img_1_ = img_1.astype(np.float32)/255.
        ###########
        diff = img_2_ - img_1_
        diff = diff*0.5 + 0.5
        # diff = (diff * 255).astype(np.uint8)
        ###########
        img = np.concatenate((img_2_, diff), axis=-1)
        return img

    def add_small_noise(self, observation):
        v = observation[0]+ random.uniform(-0.1, 0.1)
        theta_dot = observation[1]+ random.uniform(-0.001, 0.001)
        return [v, theta_dot]

    def load_training_data(self, batch_size, add_noise):
        dim_x = 2
        dim_z = 2
        dataset = pickle.load(open('KITTI_VO_dataset_v2.pkl', 'rb'))
        select = random.sample(range(0, len(dataset)), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        observation_img = []

        for idx in select:
            states_gt_save.append(dataset[idx][1])
            states_pre_save.append(dataset[idx][0])
            if add_noise == True:
                observation_save.append(self.add_small_noise(dataset[idx][2]))
            else:
                observation_save.append(dataset[idx][2])
            img = self.preprocessing(dataset[idx], False)
            observation_img.append(img)
        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        observation_img = np.array(observation_img)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [batch_size, 1, dim_x])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [batch_size, 1, dim_x])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.reshape(observation_save, [batch_size, 1, dim_z])

        observation_img = tf.convert_to_tensor(observation_img, dtype=tf.float32)

        states_pre_save = self.transform_.for_transform(states_pre_save)
        states_gt_save = self.transform_.for_transform(states_gt_save)
        observation_save = self.transform_.for_transform(observation_save)

        return states_pre_save, states_gt_save, observation_save, observation_img


    def load_testing_data(self, add_noise):
        dim_x = 2
        dim_z = 2
        dataset = pickle.load(open('KITTI_VO_test_v2.pkl', 'rb'))
        N = len(dataset)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        observation_img = []

        for idx in range(N):
            states_gt_save.append(dataset[idx][1])
            states_pre_save.append(dataset[idx][0])
            if add_noise == True:
                observation_save.append(self.add_small_noise(dataset[idx][2]))
            else:
                observation_save.append(dataset[idx][2])
            img = self.preprocessing(dataset[idx], False)
            observation_img.append(img)
        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        observation_img = np.array(observation_img)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [N, 1, dim_x])
        states_pre_save = self.transform_.for_transform(states_pre_save)

        states_pre_save = tf.reshape(states_pre_save, [N, 1, 1, dim_x])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [N, 1, dim_x])
        states_gt_save = self.transform_.for_transform(states_gt_save)

        states_gt_save = tf.reshape(states_gt_save, [N, 1, 1, dim_x])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.reshape(observation_save, [N, 1, dim_z])
        observation_save = self.transform_.for_transform(observation_save)

        observation_save = tf.reshape(observation_save, [N, 1, 1, dim_z])

        observation_img = tf.convert_to_tensor(observation_img, dtype=tf.float32)
        observation_img = tf.expand_dims(observation_img, axis=1)

        return states_pre_save, states_gt_save, observation_save, observation_img

    def load_testing_data_onebyone(self, idx, add_noise):
        dim_x = 2
        dim_z = 2
        dataset = pickle.load(open('KITTI_VO_test_v2.pkl', 'rb'))
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        observation_img = []

        N = 1

        states_gt_save.append(dataset[idx][1])
        states_pre_save.append(dataset[idx][0])
        if add_noise == True:
            observation_save.append(self.add_small_noise(dataset[idx][2]))
        else:
            observation_save.append(dataset[idx][2])
        img = self.preprocessing(dataset[idx], False)
        observation_img.append(img)

        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        observation_img = np.array(observation_img)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [N, 1, dim_x])
        states_pre_save = self.transform_.for_transform(states_pre_save)


        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [N, 1, dim_x])
        states_gt_save = self.transform_.for_transform(states_gt_save)

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.reshape(observation_save, [N, 1, dim_z])
        observation_save = self.transform_.for_transform(observation_save)

        observation_img = tf.convert_to_tensor(observation_img, dtype=tf.float32)

        return states_pre_save, states_gt_save, observation_save, observation_img


    def format_state(self, state, batch_size, num_ensemble, dim_x):
        dim_x = 2
        diag = np.ones((dim_x)).astype(np.float32) * 0.5
        # diag = np.array([10, 10, 0.1, 0.1, 0.005]).astype(np.float32)
        diag = diag.astype(np.float32)
        mean = np.zeros((dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(num_ensemble), [batch_size, num_ensemble, dim_x])
        for n in range (batch_size):
            if n == 0:
                ensemble = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
            else:
                tmp = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
                ensemble = tf.concat([ensemble, tmp], 0)
        ensemble = ensemble + Q
        state_input = (ensemble, state)
        return state_input

    def format_init_state(self, state, batch_size, num_ensemble, dim_x):
        dim_x = 2
        diag = np.ones((dim_x)).astype(np.float32) * 0.01
        # diag = np.array([1, 1, 0.1, 0.1, 0.005]).astype(np.float32)
        diag = diag.astype(np.float32)
        mean = np.zeros((dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(num_ensemble), [batch_size, num_ensemble, dim_x])
        for n in range (batch_size):
            if n == 0:
                ensemble = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
            else:
                tmp = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
                ensemble = tf.concat([ensemble, tmp], 0)
        ensemble = ensemble + Q
        state_input = (ensemble, state)
        return state_input

    def format_particle_state(self, state, batch_size, num_particles, dim_x):
        dim_x = 3
        # diag = np.ones((dim_x)).astype(np.float32) * 0.1
        diag = np.array([0.5, 0.01]).astype(np.float32)
        diag = diag.astype(np.float32)
        mean = np.zeros((dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(num_ensemble), [batch_size, num_ensemble, dim_x])
        for n in range (batch_size):
            if n == 0:
                ensemble = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
            else:
                tmp = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
                ensemble = tf.concat([ensemble, tmp], 0)
        ensemble = ensemble + Q
        state_input = (ensemble, state)
        return state_input


# DataLoader_func = DataLoader()
# add_noise = True
# states_pre_save, states_gt_save, observation_save, observation_img = DataLoader_func.load_training_data(
#     8, add_noise)
# print(states_pre_save.shape)
# print('---')
# print(states_gt_save[0])
# print('---')
# print(observation_save.shape)
# print('---')
# print(observation_img.shape)
# print('=============')

# # state = DataLoader_func.format_state(states_gt_save, 8, 32, 5)
# # print(state[0][1][0])

# states_pre_save, states_gt_save, observation_save, observation_img = DataLoader_func.load_testing_data_onebyone(1,add_noise)
# print(states_pre_save.shape)
# print('---')
# print(states_gt_save.shape)
# print('---')
# print(observation_save.shape)
# print('---')
# print(observation_img.shape)
# print('=============')

# gt = np.array(states_gt_save)
# gt = np.reshape(gt, (gt.shape[0], 2))

# x = np.linspace(1, gt.shape[0], gt.shape[0])
# fig = plt.figure(figsize=(2, 1))
# plt.subplot(2,1,1)
# plt.plot(x, gt[:,0].flatten(),color = '#e06666ff', linewidth=2.0,label = 'ground truth')
# plt.subplot(2,1,2)
# plt.plot(x, gt[:,1].flatten(),color = '#e06666ff', linewidth=2.0,label = 'ground truth')
# plt.show()













