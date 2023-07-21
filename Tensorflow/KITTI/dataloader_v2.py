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
        parameters = pickle.load(open('full_parameters.pkl', 'rb'))
        self.state_m = parameters['state_m']
        self.state_std = parameters['state_std']
        self.obs_m = parameters['obs_m']
        self.obs_std = parameters['obs_std']
        self.d_state_m = parameters['d_state_m']
        self.d_state_std = parameters['d_state_std']

    def state_transform(self, state):
        '''
        state -> [batch_size, num_ensemble, dim_x]
        '''
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_x = state.shape[2]
        state = tf.reshape(state, [batch_size * num_ensemble, dim_x])
        state = (state - self.state_m)/self.state_std
        state = tf.reshape(state, [batch_size, num_ensemble, dim_x])
        return state

    def obs_transform(self, obs):
        '''
        obs -> [batch_size, num_ensemble, dim_z]
        '''
        batch_size = obs.shape[0]
        num_ensemble = obs.shape[1]
        dim_z = obs.shape[2]
        obs = tf.reshape(obs, [batch_size * num_ensemble, dim_z])
        obs = (obs - self.obs_m)/self.obs_std
        obs = tf.reshape(obs, [batch_size, num_ensemble, dim_z])
        return obs

    def d_state_transform(self, d_state):
        '''
        d_state -> [batch_size, num_ensemble, dim_x]
        '''
        batch_size = d_state.shape[0]
        num_ensemble = d_state.shape[1]
        dim_x = d_state.shape[2]
        d_state = tf.reshape(d_state, [batch_size * num_ensemble, dim_x])
        d_state = (d_state - self.d_state_m)/self.d_state_std
        d_state = tf.reshape(d_state, [batch_size, num_ensemble, dim_x])
        return d_state

    def state_inv_transform(self, state):
        '''
        state -> [batch_size, num_ensemble, dim_x]
        '''
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_x = state.shape[2]
        state = tf.reshape(state, [batch_size * num_ensemble, dim_x])
        state = (state * self.state_std) + self.state_m 
        state = tf.reshape(state, [batch_size, num_ensemble, dim_x])
        return state

    def obs_inv_transform(self, obs):
        '''
        obs -> [batch_size, num_ensemble, dim_z]
        '''
        batch_size = obs.shape[0]
        num_ensemble = obs.shape[1]
        dim_z = obs.shape[2]
        obs = tf.reshape(obs, [batch_size * num_ensemble, dim_z])
        obs = (obs * self.obs_std) + self.obs_m 
        obs = tf.reshape(obs, [batch_size, num_ensemble, dim_z])
        return obs

    def d_state_inv_transform(self, d_state):
        '''
        d_state -> [batch_size, num_ensemble, dim_x]
        '''
        batch_size = d_state.shape[0]
        num_ensemble = d_state.shape[1]
        dim_x = d_state.shape[2]
        d_state = tf.reshape(d_state, [batch_size * num_ensemble, dim_x])
        d_state = (d_state * self.d_state_std) + self.d_state_m 
        d_state = tf.reshape(d_state, [batch_size, num_ensemble, dim_x])
        return d_state

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

    def preprocessing(self, data, train_mode, test_noise, idx):
        # img_2 = cv2.imread(self.dataset_path+data[3][1])
        # img_1 = cv2.imread(self.dataset_path+data[3][0])
        img_2 = cv2.imread(data[3][1])
        img_1 = cv2.imread(data[3][0])
        if train_mode == True:
            img_2 = cv2.flip(img_2, 1)
            img_1 = cv2.flip(img_1, 1)
        # 150, 50
        # 640, 192
        if test_noise == True:
            if idx == 1:
                img_2 = self.add_sp_noise(img_2)
                img_1 = self.add_sp_noise(img_1)
            elif idx == 2:
                img_2 = self.add_blur(img_2)
                img_1 = self.add_blur(img_1)
            elif idx == 3:
                img_2 = self.add_blackbox(img_2)
                img_1 = self.add_blackbox(img_1)
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

    def preprocessing_black(self, data, train_mode, test_noise, idx):
        # img_2 = cv2.imread(self.dataset_path+data[3][1])
        # img_1 = cv2.imread(self.dataset_path+data[3][0])
        img_2 = cv2.imread(data[3][1])
        img_1 = cv2.imread(data[3][0])
        if train_mode == True:
            img_2 = cv2.flip(img_2, 1)
            img_1 = cv2.flip(img_1, 1)
        # 150, 50
        # 640, 192
        if test_noise == True:
            if idx == 1:
                img_2 = self.add_sp_noise(img_2)
                img_1 = self.add_sp_noise(img_1)
            elif idx == 2:
                img_2 = self.add_blur(img_2)
                img_1 = self.add_blur(img_1)
            elif idx == 3:
                img_2 = self.add_blackbox(img_2)
                img_1 = self.add_blackbox(img_1)
        img_2 = cv2.resize(img_2, (640, 192), interpolation=cv2.INTER_LINEAR)
        img_1 = cv2.resize(img_1, (640, 192), interpolation=cv2.INTER_LINEAR)
        img_2_ = img_2.astype(np.float32)/255.
        img_2 = np.zeros(img_2.shape)
        img_1_ = img_1.astype(np.float32)/255.
        ###########
        diff = img_2_ - img_1_
        diff = diff*0.5 + 0.5
        # diff = (diff * 255).astype(np.uint8)
        ###########
        img = np.concatenate((img_2_, diff), axis=-1)
        return img

    def add_sp_noise(self, img):
        # Getting the dimensions of the image
        row , col, _ = img.shape
        # Randomly pick some pixels in the
        # image for coloring them white
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(8000, 10000)
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)
            # Color that pixel to white
            img[y_coord][x_coord] = 255
            
        # Randomly pick some pixels in
        # the image for coloring them black
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(8000 , 10000)
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)
            # Color that pixel to black
            img[y_coord][x_coord] = 0
        return img

    def add_blur(self, img):
        # ksize
        ksize = (15, 15)
        # Using cv2.blur() method 
        img = cv2.blur(img, ksize)
        return img

    def add_blackbox(self, img):
        start_point = (500, 100)
        # Ending coordinate, here (125, 80)
        # represents the bottom right corner of rectangle
        end_point = (start_point[0]+150, start_point[1]+150)
        # Black color in BGR
        color = (0, 0, 0)
        # Line thickness of -1 px
        # Thickness of -1 will fill the entire shape
        thickness = -1
        # Using cv2.rectangle() method
        # Draw a rectangle of black color of thickness -1 px
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        return img

    def add_small_noise(self, observation):
        v = observation[0]+ random.uniform(-0.1, 0.1)
        theta_dot = observation[1]+ random.uniform(-0.001, 0.001)
        return [v, theta_dot]
    
    def state_augment(self, pre_state, gt_state, scale):
        batch_size = pre_state.shape[0]
        num_ensemble = pre_state.shape[1]
        dim_x = pre_state.shape[2]
        if scale == True:
            value = random.uniform(-500, 500)
        else:
            value = random.uniform(-4, 4)
        add_value = np.array([value, value, 0, 0, 0])
        add = tf.reshape(tf.stack([add_value] * batch_size), [batch_size, 1, dim_x])
        add = tf.cast(add, tf.float32)
        pre_state = pre_state + add
        gt_state = gt_state + add
        return pre_state, gt_state

    def load_training_data(self, batch_size, add_noise, norm):
        dim_x = 5
        dim_z = 2
        dataset = pickle.load(open('KITTI_VO_dataset.pkl', 'rb'))
        select = random.sample(range(0, len(dataset)), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        observation_img = []
        d_state = []
        for idx in select:
            img_augment = random.uniform(0, 1)
            # if img_augment > 0.5:
            states_gt_save.append(dataset[idx][1])
            states_pre_save.append(dataset[idx][0])
            if add_noise == True:
                observation_save.append(self.add_small_noise(dataset[idx][2]))
            else:
                observation_save.append(dataset[idx][2])
            img = self.preprocessing(dataset[idx], False, False, 0)
            observation_img.append(img)
            d_state.append(dataset[idx][4])
            # else:
            #     states_gt_save.append(dataset[idx][1])
            #     states_pre_save.append(dataset[idx][0])
            #     obs = dataset[idx][2]
            #     obs[1] = -obs[1]
            #     observation_save.append(obs)
            #     img = self.preprocessing(dataset[idx], True)
            #     observation_img.append(img)
            #     d_state.append(dataset[idx][4])

        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        observation_img = np.array(observation_img)
        d_state = np.array(d_state)
        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [batch_size, 1, dim_x])
        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [batch_size, 1, dim_x])
        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.reshape(observation_save, [batch_size, 1, dim_z])
        observation_img = tf.convert_to_tensor(observation_img, dtype=tf.float32)
        if norm == True:
            states_pre_save = self.transform_.state_transform(states_pre_save)
            states_gt_save = self.transform_.state_transform(states_gt_save)
            observation_save = self.transform_.obs_transform(observation_save)
            # d_state = self.transform_.d_state_transform(d_state)
        states_pre_save, states_gt_save = self.state_augment(states_pre_save, states_gt_save, scale=False)
        d_state = states_gt_save - states_pre_save

        return states_pre_save, states_gt_save, observation_save, observation_img, d_state

    def load_testing_data_onebyone(self, idx, add_noise, norm, black):
        dim_x = 5
        dim_z = 2
        dataset = pickle.load(open('KITTI_VO_test.pkl', 'rb'))
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

        noise_type = 0
        test_noise = False
        if black == True:
            img = self.preprocessing_black(dataset[idx], False, test_noise, noise_type)
        else:
            img = self.preprocessing(dataset[idx], False, test_noise, noise_type)
        observation_img.append(img)
        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        observation_img = np.array(observation_img)
        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [N, 1, dim_x])
        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [N, 1, dim_x])
        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.reshape(observation_save, [N, 1, dim_z])
        observation_img = tf.convert_to_tensor(observation_img, dtype=tf.float32)
        observation_img = tf.expand_dims(observation_img, axis=1)
        if norm == True:
            states_pre_save = self.transform_.state_transform(states_pre_save)
            states_gt_save = self.transform_.state_transform(states_gt_save)
            observation_save = self.transform_.obs_transform(observation_save)
        d_state = states_gt_save - states_pre_save
        observation_img = tf.convert_to_tensor(observation_img, dtype=tf.float32)

        return states_pre_save, states_gt_save, observation_save, observation_img, d_state

    def load_training_data_seq(self, batch_size, window_size, norm):
        dim_x = 5
        dim_z = 2
        length_list=[4538,1098,4658,798,268,2758,1098,1098,4068,1198]
        dataset = pickle.load(open('KITTI_VO_dataset_seq.pkl', 'rb'))
        gt = []
        pre = []
        obs = []
        img_save = []
        d_state_save = []
        for i in range (batch_size):
            id_traj = random.randint(0, 9)
            start_idx = random.sample(range(0, length_list[id_traj]-window_size-1), 1)
            states_gt_save = []
            states_pre_save = []
            observation_save = []
            observation_img = []
            for idx in range (start_idx[0], start_idx[0]+window_size):
                states_gt_save.append(dataset[id_traj][idx][1])
                states_pre_save.append(dataset[id_traj][idx][0])
                observation_save.append(dataset[id_traj][idx][2])
                img = self.preprocessing(dataset[id_traj][idx], False)
                observation_img.append(img)
            states_pre_save = np.array(states_pre_save)
            states_gt_save = np.array(states_gt_save)
            observation_save = np.array(observation_save)
            observation_img = np.array(observation_img)
            # to tensor
            states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
            states_pre_save = tf.reshape(states_pre_save, [window_size, 1, dim_x])
            states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
            states_gt_save = tf.reshape(states_gt_save, [window_size, 1, dim_x])
            observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
            observation_save = tf.reshape(observation_save, [window_size, 1, dim_z])
            observation_img = tf.convert_to_tensor(observation_img, dtype=tf.float32)
            if norm == True:
                states_pre_save = self.transform_.state_transform(states_pre_save)
                states_gt_save = self.transform_.state_transform(states_gt_save)
                observation_save = self.transform_.obs_transform(observation_save)
            states_pre_save, states_gt_save = self.state_augment(states_pre_save, states_gt_save, scale=False)
            d_state = states_gt_save - states_pre_save
            gt.append(states_gt_save)
            pre.append(states_pre_save)
            obs.append(observation_save)
            img_save.append(observation_img)
            d_state_save.append(d_state)
        # to tensor
        gt = tf.convert_to_tensor(gt, dtype=tf.float32)
        pre = tf.convert_to_tensor(pre, dtype=tf.float32)
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        img_save = tf.convert_to_tensor(img_save, dtype=tf.float32)
        d_state_save = tf.convert_to_tensor(d_state_save, dtype=tf.float32)
        return pre, gt, obs, img_save, d_state_save

    def format_state(self, state, batch_size, num_ensemble, dim_x):
        dim_x = 5
        diag = np.ones((dim_x)).astype(np.float32) * 0.2
        # diag = np.array([1, 1, 0.5, 0.5, 0.005]).astype(np.float32)
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
        dim_x = 5
        diag = np.ones((dim_x)).astype(np.float32) * 0.01
        # diag = np.array([1, 1, 0.1, 0.1, 0.001]).astype(np.float32)
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
        dim_x = 5
        diag = np.ones((dim_x)).astype(np.float32) * 0.1
        diag = diag.astype(np.float32)
        mean = np.zeros((dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(num_particles), [batch_size, num_particles, dim_x])
        for n in range (batch_size):
            if n == 0:
                particles = tf.reshape(tf.stack([state[n]] * num_particles), [1, num_particles, dim_x])
            else:
                tmp = tf.reshape(tf.stack([state[n]] * num_particles), [1, num_particles, dim_x])
                particles = tf.concat([particles, tmp], 0)
        particles = particles + Q
        ud = tfp.distributions.Uniform(low=0.0, high=1.0)
        weights = tf.reshape(ud.sample(batch_size*num_particles), [batch_size,num_particles])
        w = tf.reduce_sum(weights, axis=1)
        w = tf.stack([w]*num_particles)
        w = tf.transpose(w, perm=[1,0])
        weights = weights/w
        state_input = (particles, weights, state)
        return state_input

    def format_EKF_init_state(self, state, batch_size, dim_x):
        # P = np.diag(np.array([0.01,0.01,0.01,0.1,0.1]))
        P = np.diag(np.array([1,1,1,1,1]))
        P = tf.convert_to_tensor(P, dtype=tf.float32)
        P = tf.stack([P] * batch_size)
        P = tf.reshape(P, [batch_size, dim_x, dim_x])
        state_input = (state, P)
        return state_input


# DataLoader_func = DataLoader()
# add_noise = True
# states_pre_save, states_gt_save, observation_save, observation_img, d_state = DataLoader_func.load_training_data(
#     8, add_noise, norm=True)
# print(states_pre_save[0])
# print('---')
# print(states_gt_save[0])
# print('---')
# print(observation_save.shape)
# print(observation_save[0])
# print('---')
# print(d_state.shape)
# print(d_state[0])
# print('---')
# print(observation_img.shape)
# print('=============')

# states_pre_save, states_gt_save, observation_save, observation_img, d_state = DataLoader_func.load_training_data_seq(
#     8, 5, norm=True)
# print(states_pre_save.shape)
# print('---')
# print(states_gt_save.shape)
# # print(states_gt_save[0])
# # print(states_gt_save[:,0,:,:])
# print('---')
# print(observation_save.shape)
# print('---')
# print(d_state.shape)
# print('---')
# print(observation_img.shape)
# print('=============')

# # state = DataLoader_func.format_state(states_gt_save, 8, 32, 5)
# # print(state[0][1][0])

# states_pre_save, states_gt_save, observation_save, observation_img, d_state = DataLoader_func.load_testing_data_onebyone(10, add_noise=False, norm=True)
# print(states_pre_save.shape)
# print('---')
# print(states_gt_save)
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













