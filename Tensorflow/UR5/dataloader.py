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

class DataLoader:
    # for high dimensional data
    #       state -> [time, batch_size, 1, 7] or [time, batch_size, 1, 3]
    # observation -> [time, batch_size, height, width, 3]
    def load_train_data_joint(csv_path, batch_size):
        dataset = []
        with open(csv_path,'rt')as f:
            data = csv.reader(f)
            for row in data:
                dataset.append(row)
        N = len(dataset)
        select = random.sample(range(0, N), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        for idx in select:
            arr = dataset[idx][1][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            states_pre_save.append(arr)

            arr = dataset[idx][2][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            states_gt_save.append(arr)

            img_path = './dataset'+dataset[idx][5]
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (224, 224))
            img_array = cv2.flip(img_array, 0) # flip the img vertically
            img_array = (img_array/255.0)
            observation_save.append(img_array)

        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [batch_size, 1, 7])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [batch_size, 1, 7])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        return states_pre_save, states_gt_save, observation_save

    def load_train_data_EE(csv_path, batch_size):
        dataset = []
        with open(csv_path,'rt')as f:
            data = csv.reader(f)
            for row in data:
                dataset.append(row)
        N = len(dataset)
        select = random.sample(range(0, N), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        for idx in select:
            arr = dataset[idx][3][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            states_pre_save.append(arr)

            arr = dataset[idx][4][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            states_gt_save.append(arr)

            img_path = './dataset'+dataset[idx][5]
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (224, 224))
            img_array = cv2.flip(img_array, 0) # flip the img vertically
            img_array = (img_array/255.0)
            observation_save.append(img_array)

        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [batch_size, 1, 3])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [batch_size, 1, 3])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        return states_pre_save, states_gt_save, observation_save

    def load_test_data_joint(csv_path, batch_size):
        dataset = []
        with open(csv_path,'rt')as f:
            data = csv.reader(f)
            for row in data:
                dataset.append(row)
        N = len(dataset)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        for idx in range(N):
            arr = dataset[idx][1][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            states_pre_save.append(arr)

            arr = dataset[idx][2][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            states_gt_save.append(arr)

            img_path = './dataset'+dataset[idx][5]
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (224, 224))
            img_array = cv2.flip(img_array, 0) # flip the img vertically
            img_array = (img_array/255.0)
            observation_save.append(img_array)

        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [N, batch_size, 1, 7])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [N, batch_size, 1, 7])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.expand_dims(observation_save, axis=1)
        return states_pre_save, states_gt_save, observation_save

    def load_test_data_EE(csv_path, batch_size):
        dataset = []
        with open(csv_path,'rt')as f:
            data = csv.reader(f)
            for row in data:
                dataset.append(row)
        N = len(dataset)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        for idx in range(N):
            arr = dataset[idx][3][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            states_pre_save.append(arr)

            arr = dataset[idx][4][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            states_gt_save.append(arr)

            img_path = './dataset'+dataset[idx][5]
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (224, 224))
            img_array = cv2.flip(img_array, 0) # flip the img vertically
            img_array = (img_array/255.0)
            observation_save.append(img_array)

        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [N, batch_size, 1, 3])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [N, batch_size, 1, 3])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.expand_dims(observation_save, axis=1)
        return states_pre_save, states_gt_save, observation_save

    def load_train_data_All(csv_path, batch_size):
        dataset = []
        with open(csv_path,'rt')as f:
            data = csv.reader(f)
            for row in data:
                dataset.append(row)
        N = len(dataset)
        select = random.sample(range(0, N), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        for idx in select:
            arr = dataset[idx][1][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            arr_tmp = dataset[idx][3][1:-1].split()
            arr_tmp = [eval(x) for x in arr_tmp]
            arr_tmp = np.array(arr_tmp)

            arr = np.concatenate((arr, arr_tmp), axis=None)
            states_pre_save.append(arr)

            arr = dataset[idx][2][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            arr_tmp = dataset[idx][4][1:-1].split()
            arr_tmp = [eval(x) for x in arr_tmp]
            arr_tmp = np.array(arr_tmp)
            arr = np.concatenate((arr, arr_tmp), axis=None)
            states_gt_save.append(arr)

            img_path = './dataset'+dataset[idx][5]
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (224, 224))
            img_array = cv2.flip(img_array, 0) # flip the img vertically
            img_array = (img_array/255.0)
            observation_save.append(img_array)

        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [batch_size, 1, 10])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [batch_size, 1, 10])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        return states_pre_save, states_gt_save, observation_save

    def load_test_data_All(csv_path, batch_size):
        dataset = []
        with open(csv_path,'rt')as f:
            data = csv.reader(f)
            for row in data:
                dataset.append(row)
        N = len(dataset)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        for idx in range(N):
            arr = dataset[idx][1][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            arr_tmp = dataset[idx][3][1:-1].split()
            arr_tmp = [eval(x) for x in arr_tmp]
            arr_tmp = np.array(arr_tmp)

            arr = np.concatenate((arr, arr_tmp), axis=None)
            states_pre_save.append(arr)

            arr = dataset[idx][2][1:-1].split()
            arr = [eval(x) for x in arr]
            arr = np.array(arr)
            arr_tmp = dataset[idx][4][1:-1].split()
            arr_tmp = [eval(x) for x in arr_tmp]
            arr_tmp = np.array(arr_tmp)
            arr = np.concatenate((arr, arr_tmp), axis=None)
            states_gt_save.append(arr)

            img_path = './dataset'+dataset[idx][5]
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (224, 224))
            img_array = cv2.flip(img_array, 0) # flip the img vertically
            img_array = (img_array/255.0)
            observation_save.append(img_array)

        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)

        print(states_pre_save.shape)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [N, batch_size, 1, 10])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [N, batch_size, 1, 10])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.expand_dims(observation_save, axis=1)
        return states_pre_save, states_gt_save, observation_save



    def format_state(state, batch_size, num_ensemble, dim_x):
        dim_x = dim_x
        diag = np.ones((dim_x)).astype(np.float32) * 0.1
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

    def format_init_state(state, batch_size, num_ensemble, dim_x):
        dim_x = dim_x
        diag = np.ones((dim_x)).astype(np.float32) * 0.01
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













