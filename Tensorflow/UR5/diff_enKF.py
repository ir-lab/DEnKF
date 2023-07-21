# Copyright (c) 2020 Max Planck Gesellschaft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow_probability as tfp

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''
This is the code for setting up a differentiable version of the ensemble kalman filter
The filter is trained using simulated data where we only have access to the ground truth state at each timestep
The filter is suppose to learn the process noise model Q, observation noise model R, the process model f(.) 
and the observation model h(.)
Author: Xiao -> I have made decent amount of changes to the original codebase.
'''
class ProcessModel(tf.keras.Model):
    '''
    process model is taking the state and get a prediction state and 
    calculate the jacobian matrix based on the previous state and the 
    predicted state.
    new_state = [batch_size, 1, dim_x]
            F = [batch_size, dim_x, dim_x]
    state vector 4 -> fc 32 -> fc 64 -> 2
    '''
    def __init__(self, batch_size, num_ensemble, dim_x, jacobian, rate):
        super(ProcessModel, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.jacobian = jacobian
        self.dim_x = dim_x
        self.rate = rate

    def build(self, input_shape):
        self.process_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc1')
        self.process_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc_add1')
        self.process_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc2')
        self.process_fc_add2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc_add2')
        self.process_fc3 = tf.keras.layers.Dense(
            units=self.dim_x,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc3')

    def call(self, last_state, training):
        last_state = tf.reshape(last_state, [self.batch_size * self.num_ensemble, self.dim_x])

        fc1 = self.process_fc1(last_state)
        # fc1 = tf.nn.dropout(fc1, rate=self.rate)
        fcadd1 = self.process_fc_add1(fc1)
        # fcadd1 = tf.nn.dropout(fcadd1, rate=self.rate)
        fc2 = self.process_fc2(fcadd1)
        fc2 = tf.nn.dropout(fc2, rate=self.rate)
        fcadd2 = self.process_fc_add2(fc2)
        fcadd2 = tf.nn.dropout(fcadd2, rate=self.rate)
        update = self.process_fc3(fcadd2)

        new_state = last_state + update
        new_state = tf.reshape(new_state, [self.batch_size, self.num_ensemble, self.dim_x])

        return new_state

class BayesianProcessModel(tf.keras.Model):
    '''
    process model is taking the state and get a prediction state and 
    calculate the jacobian matrix based on the previous state and the 
    predicted state.
    new_state = [batch_size, 1, dim_x]
            F = [batch_size, dim_x, dim_x]
    state vector 4 -> fc 32 -> fc 64 -> 2
    '''
    def __init__(self, batch_size, num_ensemble, dim_x, jacobian, rate):
        super(BayesianProcessModel, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.jacobian = jacobian
        self.dim_x = dim_x
        self.rate = rate

    def build(self, input_shape):
        self.process_fc1 = tfp.layers.DenseFlipout(
            units=32,
            activation = tf.nn.relu,
            name='process_fc1')
        self.process_fc_add1 = tfp.layers.DenseFlipout(
            units=64,
            activation=tf.nn.relu,
            name='process_fc_add1')
        self.process_fc2 = tfp.layers.DenseFlipout(
            units=64,
            activation=tf.nn.relu,
            name='process_fc2')
        self.process_fc_add2 = tfp.layers.DenseFlipout(
            units=32,
            activation=tf.nn.relu,
            name='process_fc_add2')
        self.process_fc3 = tfp.layers.DenseFlipout(
            units=self.dim_x,
            activation=None,
            name='process_fc3')

    def call(self, last_state, training):
        last_state = tf.reshape(last_state, [self.batch_size * self.num_ensemble, self.dim_x])

        fc1 = self.process_fc1(last_state)
        fcadd1 = self.process_fc_add1(fc1)
        fc2 = self.process_fc2(fcadd1)
        # fc2 = tf.nn.dropout(fc2, rate=self.rate)
        fcadd2 = self.process_fc_add2(fc2)
        # fcadd2 = tf.nn.dropout(fcadd2, rate=self.rate)
        update = self.process_fc3(fcadd2)

        new_state = last_state + update
        new_state = tf.reshape(new_state, [self.batch_size, self.num_ensemble, self.dim_x])

        return new_state

class addAction(tf.keras.Model):
    '''
    action models serves in the prediction step and it will be added to predicted state before the 
    updating steps. In this toy example.
     State: [batch_size, 1, dim_x]
         B: [batch_size, dim_x, 2]
    action: [batch_size, 1, 2]
    '''
    def __init__(self, batch_size, dim_x):
        super(addAction, self).__init__()
        self.batch_size = batch_size
        self.dim_x = dim_x

    def call(self, state, action, training):
        DT = 0.1
        for i in range (self.batch_size):
            if i == 0:
                B_tmp = np.array([
                    [DT * np.cos(state[i][0][2]), 0],
                    [DT * np.sin(state[i][0][2]), 0],
                    [0.0, DT],
                    [1.0, 0.0]])
                B = tf.reshape(B_tmp, [1,self.dim_x,2])
            else:
                B_tmp = np.array([
                    [DT * np.cos(state[i][0][2]), 0],
                    [DT * np.sin(state[i][0][2]), 0],
                    [0.0, DT],
                    [1.0, 0.0]])
                B_tmp = tf.reshape(B_tmp, [1,self.dim_x,2])
                B = tf.concat([B, B_tmp], axis = 0)
        B = tf.cast(B, tf.float32)
        state = state + tf.transpose(tf.matmul(B, tf.transpose(action, perm=[0,2,1])), perm=[0,2,1])
        return state

class ObservationModel(tf.keras.Model):
    '''
    process model is taking the state and get a prediction state and 
    calculate the jacobian matrix based on the previous state and the 
    predicted state.
    new_state = [batch_size, 1, dim_x]
            F = [batch_size, dim_x, dim_x]
    state vector 4 -> fc 32 -> fc 64 -> 2
    '''
    def __init__(self, batch_size, num_ensemble, dim_x, dim_z, jacobian):
        super(ObservationModel, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.jacobian = jacobian
        self.dim_x = dim_x
        self.dim_z = dim_z

    def build(self, input_shape):
        self.observation_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc1')
        self.observation_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc_add1')
        self.observation_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc2')
        self.observation_fc_add2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc_add2')
        self.observation_fc3 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc3')

    def call(self, state, training, learn):
        state = tf.reshape(state, [self.batch_size* self.num_ensemble, 1, self.dim_x])
        if learn == False:
            H = tf.concat(
                [tf.tile(np.array([[[1, 0, 0, 0, 0]]], dtype=np.float32),
                         [self.batch_size* self.num_ensemble, 1, 1]),
                 tf.tile(np.array([[[0, 1, 0, 0, 0]]], dtype=np.float32),
                         [self.batch_size* self.num_ensemble, 1, 1])], axis=1)
            z_pred = tf.matmul(H, tf.transpose(state, perm=[0,2,1]))
            Z_pred = tf.transpose(z_pred, perm=[0,2,1])
            z_pred = tf.reshape(z_pred, [self.batch_size, self.num_ensemble, self.dim_z])
        else:
            fc1 = self.observation_fc1(state)
            fcadd1 = self.observation_fc_add1(fc1)
            fc2 = self.observation_fc2(fcadd1)
            fcadd2 = self.observation_fc_add2(fc2)
            z_pred = self.observation_fc3(fcadd2)
            z_pred = tf.reshape(z_pred, [self.batch_size, self.num_ensemble, self.dim_z])

        return z_pred


class SensorModel(tf.keras.Model):
    '''
    sensor model is used for modeling H with given states to get observation z
    it is not required for this model to take states only, if the obervation is 
    an image or higher dimentional tensor, it is supposed to learn a lower demention
    representation from the observation space.
    observation = [batch_size, dim_z]
    encoding = [batch_size, dim_fc2]
    '''
    def __init__(self, batch_size, dim_z):
        super(SensorModel, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z

    def build(self, input_shape):
        self.sensor_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc1')
        self.sensor_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc_add1')
        self.sensor_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc2')
        self.sensor_fc_add2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc_add2')
        self.sensor_fc3 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc3')

    def call(self, state, training, learn):
        if learn == True:
            fc1 = self.sensor_fc1(state)
            fcadd1 = self.sensor_fc_add1(fc1)
            fc2 = self.sensor_fc2(fcadd1)
            fcadd2 = self.sensor_fc_add2(fc2)
            observation = self.sensor_fc3(fcadd2)
            encoding = fcadd2
        else:
            observation = state
            encoding = state

        return observation, encoding

class BayesianSensorModel(tf.keras.Model):
    '''
    sensor model is used for modeling H with given states to get observation z
    it is not required for this model to take states only, if the obervation is 
    an image or higher dimentional tensor, it is supposed to learn a lower demention
    representation from the observation space.
    observation = [batch_size, dim_z]
    encoding = [batch_size, dim_fc2]
    '''
    def __init__(self, batch_size, num_ensemble, dim_z):
        super(BayesianSensorModel, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.num_ensemble = num_ensemble

    def build(self, input_shape):
        # bayesian neural networks
        self.bayes_sensor_fc1 = tfp.layers.DenseFlipout(
            units=64,
            activation=tf.nn.relu,
            name='bayes_sensor_fc1')
        self.bayes_sensor_fc2 = tfp.layers.DenseFlipout(
            units=32,
            activation=tf.nn.relu,
            name='bayes_sensor_fc2')
        self.bayes_sensor_fc3 = tfp.layers.DenseFlipout(
            units=32,
            activation=tf.nn.relu,
            name='bayes_sensor_fc3')
        self.bayes_sensor_fc4 = tfp.layers.DenseFlipout(
            units=self.dim_z,
            activation=None,
            name='bayes_sensor_fc4')

    def call(self, state, training, learn):
        if learn == True:
            inputs = state
            num_feature = inputs.shape[1]
            # expand to ensembles
            for i in range (self.batch_size):
                if i == 0:
                    inputs_z = tf.reshape(tf.stack([inputs[i]] * self.num_ensemble), [1, self.num_ensemble, num_feature])
                else:
                    tmp = tf.reshape(tf.stack([inputs[i]] * self.num_ensemble), [1, self.num_ensemble, num_feature])
                    inputs_z = tf.concat([inputs_z, tmp], 0)

            # make sure the ensemble shape matches
            inputs_z = tf.reshape(inputs_z, [self.batch_size * self.num_ensemble, num_feature])

            fc1 = self.bayes_sensor_fc1(inputs_z)
            fc2 = self.bayes_sensor_fc2(fc1)
            fcadd2 = self.bayes_sensor_fc3(fc2)
            observation = self.bayes_sensor_fc4(fcadd2)
            encoding = fcadd2

            observation = tf.reshape(observation, [self.batch_size, self.num_ensemble, self.dim_z])
            observation_m = tf.reduce_mean(observation, axis = 1)

            encoding = tf.reshape(encoding, [self.batch_size, self.num_ensemble, 32])
            encoding = tf.reduce_mean(encoding, axis = 1)
        else:
            observation = state
            encoding = state

        return observation, observation_m, encoding

class BayesianImageSensorModel(tf.keras.Model):
    '''
    sensor model is used for modeling H with given states to get observation z
    it is not required for this model to take states only, if the obervation is 
    an image or higher dimentional tensor, it is supposed to learn a lower demention
    representation from the observation space.
    observation = [batch_size, dim_z]
    encoding = [batch_size, dim_fc2]
    '''
    def __init__(self, batch_size, num_ensemble, dim_z):
        super(BayesianImageSensorModel, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.num_ensemble = num_ensemble

    def build(self, input_shape):
        self.sensor_conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv1')

        self.sensor_conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=5,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv2')

        self.sensor_conv3 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv3')

        self.sensor_conv4 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3,
            strides=[1, 1],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv4')

        self.flatten = tf.keras.layers.Flatten()

        # bayesian neural networks
        self.bayes_sensor_fc1 = tfp.layers.DenseFlipout(
            units=64,
            activation=tf.nn.relu,
            name='bayes_sensor_fc1')
        self.bayes_sensor_fc2 = tfp.layers.DenseFlipout(
            units=32,
            activation=tf.nn.relu,
            name='bayes_sensor_fc2')
        self.bayes_sensor_fc3 = tfp.layers.DenseFlipout(
            units=32,
            activation=tf.nn.relu,
            name='bayes_sensor_fc3')
        self.bayes_sensor_fc4 = tfp.layers.DenseFlipout(
            units=self.dim_z,
            activation=None,
            name='bayes_sensor_fc4')

    def call(self, image, training, learn):
        if learn == True:
            conv1 = self.sensor_conv1(image)
            conv1 = tf.nn.max_pool2d(conv1, 2, 2, padding='SAME')
            conv2 = self.sensor_conv2(conv1)
            conv2 = tf.nn.max_pool2d(conv2, 2, 2, padding='SAME')
            conv3 = self.sensor_conv3(conv2)
            conv3 = tf.nn.max_pool2d(conv3, 2, 2, padding='SAME')
            conv4 = self.sensor_conv4(conv3)

            inputs = self.flatten(conv4)
            num_feature = inputs.shape[1]

            # expand to ensembles
            for i in range (self.batch_size):
                if i == 0:
                    inputs_z = tf.reshape(tf.stack([inputs[i]] * self.num_ensemble), [1, self.num_ensemble, num_feature])
                else:
                    tmp = tf.reshape(tf.stack([inputs[i]] * self.num_ensemble), [1, self.num_ensemble, num_feature])
                    inputs_z = tf.concat([inputs_z, tmp], 0)

            # make sure the ensemble shape matches
            inputs_z = tf.reshape(inputs_z, [self.batch_size * self.num_ensemble, num_feature])

            fc1 = self.bayes_sensor_fc1(inputs_z)
            fc2 = self.bayes_sensor_fc2(fc1)
            fcadd2 = self.bayes_sensor_fc3(fc2)
            observation = self.bayes_sensor_fc4(fcadd2)
            encoding = fcadd2

            observation = tf.reshape(observation, [self.batch_size, self.num_ensemble, self.dim_z])
            observation_m = tf.reduce_mean(observation, axis = 1)

            encoding = tf.reshape(encoding, [self.batch_size, self.num_ensemble, 32])
            encoding = tf.reduce_mean(encoding, axis = 1)
        else:
            observation = state
            encoding = state

        return observation, observation_m, encoding


class ImageSensorModel(tf.keras.Model):
    '''
    sensor model is used for modeling H with given states to get observation z
    it is not required for this model to take states only, if the obervation is 
    an image or higher dimentional tensor, it is supposed to learn a lower demention
    representation from the observation space.
    observation = [batch_size, dim_z]
    encoding = [batch_size, dim_fc2]
    '''
    def __init__(self, batch_size, dim_z):
        super(ImageSensorModel, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z

    def build(self, input_shape):
        self.sensor_conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv1')
        self.sensor_conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv2')

        self.sensor_conv3 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv3')

        self.flatten = tf.keras.layers.Flatten()

        self.sensor_fc1 = tf.keras.layers.Dense(
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc1')
        self.sensor_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc_add1')
        self.sensor_fc2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc2')
        self.sensor_fc_add2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc_add2')
        self.sensor_fc3 = tf.keras.layers.Dense(
            units=self.dim_z,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc3',
            activation=None)

    def call(self, image, training, learn):
        if learn == True:
            conv1 = self.sensor_conv1(image)
            conv1 = tf.nn.max_pool2d(conv1, 2, 2, padding='SAME')
            conv2 = self.sensor_conv2(conv1)
            conv2 = tf.nn.max_pool2d(conv2, 2, 2, padding='SAME')
            conv3 = self.sensor_conv3(conv2)

            inputs = self.flatten(conv3)

            fc1 = self.sensor_fc1(inputs)
            fcadd1 = self.sensor_fc_add1(fc1)
            fc2 = self.sensor_fc2(fcadd1)
            fcadd2 = self.sensor_fc_add2(fc2)
            observation = self.sensor_fc3(fcadd2)
            encoding = fcadd2
        else:
            observation = state
            encoding = state

        return observation, encoding

class ProcessNoise(tf.keras.Model):
    '''
    Noise model is asuming the noise to be heteroscedastic
    The noise is not constant at each step
    The fc neural network is designed for learning the diag(Q)
    Q = [batch_size, dim_x, dim_x]
    i.e., 
    if the state has 4 inputs
    state vector 4 -> fc 32 -> fc 64 -> 4
    the result is the diag of Q where Q is a 4x4 matrix
    '''
    def __init__(self, batch_size, num_ensemble, dim_x, q_diag):
        super(ProcessNoise, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.q_diag = q_diag

    def build(self, input_shape):
        constant = np.ones(self.dim_x)* 1e-3
        init = np.sqrt(np.square(self.q_diag) - constant)
        self.fixed_process_noise_bias = self.add_weight(
            name = 'fixed_process_noise_bias',
            shape = [self.dim_x],
            regularizer = tf.keras.regularizers.l2(l=1e-3),
            initializer = tf.constant_initializer(constant))
        self.process_noise_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_noise_fc1')
        self.process_noise_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_noise_fc_add1')
        self.process_noise_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_noise_fc2')
        self.process_noise_fc3 = tf.keras.layers.Dense(
            units=self.dim_x,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_noise_fc3')
        self.learned_process_noise_bias = self.add_weight(
            name = 'learned_process_noise_bias',
            shape = [self.dim_x],
            regularizer = tf.keras.regularizers.l2(l=1e-3),
            initializer = tf.constant_initializer(init))

    def call(self, state, training, learn):
        if learn == True:
            fc1 = self.process_noise_fc1(state)
            fcadd1 = self.process_noise_fc_add1(fc1)
            fc2 = self.process_noise_fc2(fcadd1)
            diag = self.process_noise_fc3(fc2)
            diag = tf.square(diag + self.learned_process_noise_bias)
        else:
            diag = tf.square(self.learned_process_noise_bias)
            diag = tf.stack([diag] * (self.batch_size))

        diag = diag + self.fixed_process_noise_bias
        mean = np.zeros((self.dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * self.batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(self.num_ensemble), [self.batch_size, self.num_ensemble, self.dim_x])

        return Q, diag


class ObservationNoise(tf.keras.Model):
    '''
    Noise model is asuming the noise to be heteroscedastic
    The noise is not constant at each step
    inputs: an intermediate representation of the raw observation
    denoted as encoding 
    R = [batch_size, dim_z, dim_z]
    The fc neural network is designed for learning the diag(R)
    i.e., 
    if the state has 4 inputs, the encoding has size 64,
    observation vector z is with size 2, the R has the size
    2 + (64 -> fc 2 -> 2) + fixed noise,
    the result is the diag of R where R is a 2x2 matrix
    '''
    def __init__(self, batch_size, num_ensemble, dim_z, r_diag, jacobian):
        super(ObservationNoise, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.jacobian = jacobian
        self.dim_z = dim_z
        self.r_diag = r_diag

    def build(self, input_shape):
        constant = np.ones(self.dim_z)* 1e-3
        init = np.sqrt(np.square(self.r_diag) - constant)
        self.fixed_observation_noise_bias = self.add_weight(
            name = 'fixed_observation_noise_bias',
            shape = [self.dim_z],
            regularizer = tf.keras.regularizers.l2(l=1e-3),
            initializer = tf.constant_initializer(constant))

        self.observation_noise_fc1 = tf.keras.layers.Dense(
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_noise_fc1')

        self.observation_noise_fc2 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_noise_fc2')

        self.learned_observation_noise_bias = self.add_weight(
            name = 'learned_observation_noise_bias',
            shape = [self.dim_z],
            regularizer = tf.keras.regularizers.l2(l=1e-3),
            initializer = tf.constant_initializer(init))

    def call(self, inputs, training, learn):
        if learn == True:
            diag = self.observation_noise_fc1(inputs)
            diag = self.observation_noise_fc2(diag)
            diag = tf.square(diag + self.learned_observation_noise_bias)
        else:
            diag = tf.square(self.learned_observation_noise_bias)
            diag = tf.stack([diag] * (self.batch_size))

        diag = diag + self.fixed_observation_noise_bias
        R = tf.linalg.diag(diag)
        R = tf.reshape(R, [self.batch_size, self.dim_z, self.dim_z])
        diag = tf.reshape(diag, [self.batch_size, self.dim_z])

        return R, diag


class getloss():
    def _mse(self, diff):
        """
        Returns the mean squared error of diff = label - pred plus their
        euclidean distance (dist)
        Parameters
        ----------
        diff : tensor
            difference between label and prediction
        reduce_mean : bool, optional
            if true, return the mean errors over the complete tensor. The
            default is False.
        Returns
        -------
        loss : tensor
            the mean squared error
        dist : tensor
            the euclidean distance
        """
        diff_a = tf.expand_dims(diff, axis=-1)
        diff_b = tf.expand_dims(diff, axis=-2)

        loss = tf.matmul(diff_b, diff_a)

        # the loss needs to be finite and positive
        loss = tf.where(tf.math.is_finite(loss), loss,
                        tf.ones_like(loss)*1e20)
        loss = tf.where(tf.greater_equal(loss, 0), loss,
                        tf.ones_like(loss)*1e20)

        loss = tf.squeeze(loss, axis=-1)
        dist = tf.sqrt(loss)

        loss = tf.reduce_mean(loss)
        dist = tf.reduce_mean(dist)

        loss = dist + loss

        return loss

class utils:
    def __init__(self):
        super(utils, self).__init__()
        self.scale = 1
    ###########################################################################
    # convenience functions for ensuring stability

    ###########################################################################
    def _condition_number(self, s):
        """
        Compute the condition number of a matrix based on its eigenvalues s
        Parameters
        ----------
        s : tensor
            the eigenvalues of a matrix
        Returns
        -------
        r_corrected : tensor
            the condition number of the matrix
        """
        r = s[..., 0] / s[..., -1]

        # Replace NaNs in r with infinite
        r_nan = tf.math.is_nan(r)
        r_inf = tf.fill(tf.shape(r), tf.constant(np.Inf, r.dtype))
        r_corrected = tf.where(r_nan, r_inf, r)

        return r_corrected

    def _is_invertible(self, s, epsilon=1e-6):
        """
        Check if a matrix is invertible based on its eigenvalues s
        Parameters
        ----------
        s : tensor
            the eigenvalues of a matrix
        epsilon : float, optional
            threshold for the condition number
        Returns
        -------
        invertible : tf.bool tensor
            true if the matrix is invertible
        """
        # "c"
        # Epsilon may be smaller with tf.float64
        eps_inv = tf.cast(1. / epsilon, s.dtype)
        cond_num = self._condition_number(s)
        invertible = tf.logical_and(tf.math.is_finite(cond_num),
                                    tf.less(cond_num, eps_inv))
        return invertible

    def _make_valid(self, covar):
        """
        Trys to make a possibly degenerate covariance valid by
          - replacing nans and infs with high values/zeros
          - making the matrix symmetric
          - trying to make the matrix invertible by adding small offsets to
            the smallest eigenvalues
        Parameters
        ----------
        covar : tensor
            a covariance matrix that is possibly degenerate
        Returns
        -------
        covar_valid : tensor
            a covariance matrix that is hopefully valid
        """
        # eliminate nans and infs (replace them with high values on the
        # diagonal and zeros else)
        bs = covar.get_shape()[0]
        dim = covar.get_shape()[-1]
        covar = tf.where(tf.math.is_finite(covar), covar,
                         tf.eye(dim, batch_shape=[bs])*1e6)

        # make symmetric
        covar = (covar + tf.linalg.matrix_transpose(covar)) / 2.

        # add a bit of noise to the diagonal of covar to prevent
        # nans in the gradient of the svd
        noise = tf.random.uniform(covar.get_shape().as_list()[:-1], minval=0,
                                  maxval=0.001/self.scale**2)
        s, u, v = tf.linalg.svd(covar + tf.linalg.diag(noise))
        # test if the matrix is invertible
        invertible = self._is_invertible(s)
        # test if the matrix is positive definite
        pd = tf.reduce_all(tf.greater(s, 0), axis=-1)

        # try making a valid version of the covariance matrix by ensuring that
        # the minimum eigenvalue is at least 1e-4/self.scale
        min_eig = s[..., -1:]
        eps = tf.tile(tf.maximum(1e-4/self.scale - min_eig, 0),
                      [1, s.get_shape()[-1] ])
        covar_invertible = tf.matmul(u, tf.matmul(tf.linalg.diag(s + eps), v,
                                                  adjoint_b=True))

        # if the covariance matrix is valid, leave it as is, else replace with
        # the modified variant
        covar_valid = tf.where(tf.logical_and(invertible, pd)[:, None, None],
                               covar, covar_invertible)

        # make symmetric again
        covar_valid = \
            (covar_valid + tf.linalg.matrix_transpose(covar_valid)) / 2.

        return covar_valid
    ###########################################################################

class bayesiantransition(tf.keras.Model):
    def __init__(self, batch_size, num_ensemble, dropout_rate,**kwargs):

        super(bayesiantransition, self).__init__(**kwargs)

        # initialization
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        
        self.dim_x = 4

        self.jacobian = True

        self.dropout_rate = dropout_rate

        # learned process model
        self.bayesian_model = BayesianProcessModel(self.batch_size, self.num_ensemble, self.dim_x, self.jacobian, self.dropout_rate)

    def call(self, input_states):

        state_old, m_state = input_states

        state_old = tf.reshape(state_old, [self.batch_size, self.num_ensemble, self.dim_x])

        m_state = tf.reshape(m_state, [self.batch_size, self.dim_x])

        # get prediction and noise of next state
        training = True
        state_pred = self.bayesian_model(state_old, training)

        # the ensemble state mean
        m_state = tf.reduce_mean(state_pred, axis = 1)

        ensemble = tf.reshape(state_pred, [self.batch_size, self.num_ensemble, self.dim_x])

        m_state = tf.reshape(m_state, [self.batch_size, 1, self.dim_x])

        # tuple structure of updated state
        output = (ensemble, m_state)

        return output



# Xiao's version
class enKFMLP(tf.keras.Model):
    def __init__(self, batch_size, num_ensemble, dropout_rate,**kwargs):
        super(enKFMLP, self).__init__()

        # initialization
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        
        self.dim_x = 10
        self.dim_z = 10

        self.jacobian = True

        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.1
        self.r_diag = self.r_diag.astype(np.float32)

        self.dropout_rate = dropout_rate

        self.bayesian_process_model = BayesianProcessModel(self.batch_size, self.num_ensemble, self.dim_x, self.jacobian, self.dropout_rate)

        # learned observation model
        self.observation_model = ObservationModel(self.batch_size, self.num_ensemble, self.dim_x, self.dim_z, self.jacobian)

        # learned observation noise
        self.observation_noise_model = ObservationNoise(self.batch_size, self.num_ensemble, self.dim_z, self.r_diag, self.jacobian)

        # learned sensor model
        self.sensor_model = BayesianImageSensorModel(self.batch_size, self.num_ensemble, self.dim_z)

        self.utils_ = utils()

    def call(self, inputs, states):
        # decompose inputs and states
        raw_sensor = inputs

        state_old, m_state = states

        state_old = tf.reshape(state_old, [self.batch_size, self.num_ensemble, self.dim_x])

        m_state = tf.reshape(m_state, [self.batch_size, self.dim_x])


        # get prediction and noise of next state
        training = True
        state_pred = self.bayesian_process_model(state_old, training)


        # update step
        # get predicted observations
        learn = True
        H_X = self.observation_model(state_pred, training, learn)

        # get the emsemble mean of the observations
        m = tf.reduce_mean(H_X, axis = 1)
        for i in range (self.batch_size):
            if i == 0:
                mean = tf.reshape(tf.stack([m[i]] * self.num_ensemble), [self.num_ensemble, self.dim_z])
            else:
                tmp = tf.reshape(tf.stack([m[i]] * self.num_ensemble), [self.num_ensemble, self.dim_z])
                mean = tf.concat([mean, tmp], 0)

        mean = tf.reshape(mean, [self.batch_size, self.num_ensemble, self.dim_z])
        H_A = H_X - mean

        final_H_A = tf.transpose(H_A, perm=[0,2,1])
        final_H_X = tf.transpose(H_X, perm=[0,2,1])

        # get sensor reading
        ensemble_z, z, encoding = self.sensor_model(raw_sensor, training, learn = True)

        # get observation noise
        R, diag_R = self.observation_noise_model(encoding, training, True)


        # the measurement y
        y = ensemble_z
        y = tf.transpose(y, perm=[0,2,1])

        # calculated innovation matrix s
        innovation = (1/(self.num_ensemble -1)) * tf.matmul(final_H_A,  H_A) + R

        # A matrix
        m_A = tf.reduce_mean(state_pred, axis = 1)
        for i in range (self.batch_size):
            if i == 0:
                mean_A = tf.reshape(tf.stack([m_A[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_x])
            else:
                tmp = tf.reshape(tf.stack([m_A[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_x])
                mean_A = tf.concat([mean_A, tmp], 0)
        A = state_pred - mean_A
        A = tf.transpose(A, perm = [0,2,1])

        try:
            innovation_inv = tf.linalg.inv(innovation)
        except:
            innovation = self.utils_._make_valid(innovation)
            innovation_inv = tf.linalg.inv(innovation)

        # calculating Kalman gain
        K = (1/(self.num_ensemble -1)) * tf.matmul(tf.matmul(A, H_A), innovation_inv)

        # update state of each ensemble
        y_bar = y - final_H_X
        state_new = state_pred +  tf.transpose(tf.matmul(K, y_bar), perm=[0,2,1])

        # the ensemble state mean
        m_state_new = tf.reduce_mean(state_new, axis = 1)

        m_state_new = tf.reshape(m_state_new, [self.batch_size, 1, self.dim_x])

        m_state_pred = tf.reduce_mean(state_pred, axis = 1)

        m_state_pred = tf.reshape(m_state_pred, [self.batch_size, 1, self.dim_x])

        z = tf.reshape(z, [self.batch_size, 1, self.dim_z])

        # tuple structure of updated state
        output = (state_new, m_state_new, m_state_pred, z)

        return output


