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
from dataloader_v2 import transform

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''
This is the code for setting up a differentiable version of the ensemble kalman filter
The filter is trained using simulated/real data where we have access to the ground truth state at each timestep
The filter is suppose to learn the process noise model Q, observation noise model R, the process model f(.) 
and the observation model h(.)
Author: Xiao Liu -> I have made decent amount of changes to the original codebase.
'''

class BayesianProcessModel(tf.keras.Model):
    '''
    process model is taking the state and get a distribution of a prediction state,
    which is represented as ensemble.
    new_state = [batch_size, num_ensemble, dim_x]
    state vector dim_x -> fc 32 -> fc 64 -> fc 32 -> dim_x
    '''
    def __init__(self, batch_size, num_ensemble, dim_x):
        super(BayesianProcessModel, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        # self.state_transform_ = state_transform()

    def build(self, input_shape):
        self.process_fc1 = tfp.layers.DenseFlipout(
            units=64,
            activation = tf.nn.relu,
            name='process_fc1')
        self.process_fc_add1 = tfp.layers.DenseFlipout(
            units=128,
            activation=tf.nn.relu,
            name='process_fc_add1')
        self.process_fc2 = tfp.layers.DenseFlipout(
            units=128,
            activation=tf.nn.relu,
            name='process_fc2')
        self.process_fc_add2 = tfp.layers.DenseFlipout(
            units=64,
            activation=tf.nn.relu,
            name='process_fc_add2')
        self.process_fc3 = tfp.layers.DenseFlipout(
            units=self.dim_x,
            activation=None,
            name='process_fc3')

    def call(self, last_state):
        last_state = tf.reshape(last_state, [self.batch_size * self.num_ensemble, self.dim_x])
        fc1 = self.process_fc1(last_state)
        fcadd1 = self.process_fc_add1(fc1)
        fc2 = self.process_fc2(fcadd1) 
        fcadd2 = self.process_fc_add2(fc2)
        update = self.process_fc3(fcadd2)
        new_state = last_state+update
        new_state = tf.reshape(new_state, [self.batch_size, self.num_ensemble, self.dim_x])
        return new_state

class ObservationModel(tf.keras.Model):
    '''
    observation model takes a predicted state and map it to the observation space
    state vector dim_x -> fc 32 -> fc 64 -> fc 64 -> fc 32 -> dim_z
    '''
    def __init__(self, batch_size, num_ensemble, dim_x, dim_z):
        super(ObservationModel, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z

    def build(self, input_shape):
        self.observation_fc1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc1')
        self.observation_fc2 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc2')
        self.observation_fc3 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc3')
        self.observation_fc4 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc4')
        self.observation_fc5 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc5')

    def call(self, state):
        state = tf.reshape(state, [self.batch_size* self.num_ensemble, self.dim_x])
        fc1 = self.observation_fc1(state)
        fc2 = self.observation_fc2(fc1)
        fc3 = self.observation_fc3(fc2)
        fc4 = self.observation_fc4(fc3)
        z_pred = self.observation_fc5(fc4)
        z_pred = tf.reshape(z_pred, [self.batch_size, self.num_ensemble, self.dim_z])
        return z_pred


class SensorModel(tf.keras.Model):
    '''
    sensor model is used for learning a representation of the current observation,
    the representation can be explainable or latent.  
    observation = [batch_size, img_h, img_w, channel]
    encoding = [batch_size, dim_fc3]
    '''
    def __init__(self, batch_size, num_ensemble, dim_z):
        super(SensorModel, self).__init__()
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
            filters=128, kernel_size=5,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv2')

        self.sensor_conv3 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=5,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv3')

        self.sensor_conv4 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=5,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv4')

        self.sensor_conv5 = tf.keras.layers.Conv2D(
            filters=512, kernel_size=3,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv5')

        self.flatten = tf.keras.layers.Flatten()

        # bayesian neural networks
        self.sensor_fc1 = tf.keras.layers.Dense(
            units=1024,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc1')
        self.sensor_fc2 = tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc2')
        self.sensor_fc3 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc3')

    def call(self, image):
        conv1 = self.sensor_conv1(image)
        conv1 = tf.keras.layers.BatchNormalization(axis = -1)(conv1)
        conv2 = self.sensor_conv2(conv1)
        conv2 = tf.keras.layers.BatchNormalization(axis = -1)(conv2)
        conv3 = self.sensor_conv3(conv2)
        conv3 = tf.keras.layers.BatchNormalization(axis = -1)(conv3)
        conv4 = self.sensor_conv4(conv3)
        conv4 = tf.keras.layers.BatchNormalization(axis = -1)(conv4)
        conv5 = self.sensor_conv5(conv4)
        conv5 = tf.keras.layers.BatchNormalization(axis = -1)(conv5)
        conv5 = tf.nn.dropout(conv5, rate=0.3)
        inputs = self.flatten(conv5)
        fc1 = self.sensor_fc1(inputs)
        fc2 = self.sensor_fc2(fc1)
        observation = self.sensor_fc3(fc2)
        observation = tf.reshape(observation, [self.batch_size, 1, self.dim_z])
        return observation

class ObservationNoise(tf.keras.Model):
    '''
    observation noise model is used for estimating the aleatoric noise.
    inputs: an intermediate representation of the raw observation
    denoted as encoding.
    R = [batch_size, dim_z, dim_z]
    The fc neural network is designed for learning the diag(R)
    i.e., 
    if the state has 4 inputs, the encoding has size 64,
    observation vector z is with size 2, the R has the size
    2 + (64 -> fc 2 -> 2) + fixed noise,
    the result is the diag of R where R is a 2x2 matrix
    '''
    def __init__(self, batch_size, num_ensemble, dim_z, r_diag):
        super(ObservationNoise, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
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
            units=32,
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

    def call(self, inputs):
        inputs = tf.reshape(inputs, [self.batch_size, self.dim_z])
        diag = self.observation_noise_fc1(inputs)
        diag = self.observation_noise_fc2(diag)
        diag = tf.square(diag + self.learned_observation_noise_bias)
        diag = diag + self.fixed_observation_noise_bias
        R = tf.linalg.diag(diag)
        R = tf.reshape(R, [self.batch_size, self.dim_z, self.dim_z])
        diag = tf.reshape(diag, [self.batch_size, self.dim_z])
        return R, diag

class StateDecoderModel(tf.keras.Model):
    '''
    sensor model is used for learning a representation of the current observation,
    the representation can be explainable or latent.  
    observation = [batch_size, img_h, img_w, channel]
    encoding = [batch_size, dim_fc3]
    '''
    def __init__(self, batch_size, num_ensemble, dim_x):
        super(StateDecoderModel, self).__init__()
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.num_ensemble = num_ensemble

    def build(self, input_shape):
        self.decode_fc1 = tf.keras.layers.Dense(
            units=1024,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='decode_fc1')
        self.decode_fc2 = tf.keras.layers.Dense(
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='decode_fc2')
        self.decode_fc3 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='decode_fc3')
        self.decode_fc4 = tf.keras.layers.Dense(
            units=5,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='decode_fc4')

    def call(self, state):
        state = tf.reshape(state, [self.batch_size* self.num_ensemble, self.dim_x])
        fc1 = self.decode_fc1(state)
        fc2 = self.decode_fc2(fc1)
        fc3 = self.decode_fc3(fc2)
        out = self.decode_fc4(fc3)
        out = tf.reshape(out, [self.batch_size, self.num_ensemble, 5])
        return out


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

class StandaloneModel(tf.keras.Model):
    def __init__(self, batch_size, num_ensemble, **kwargs):
        super(StandaloneModel, self).__init__(**kwargs)
        # initialization
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.dim_x = 200
        self.dim_z = 200

        # learned sensor model
        self.sensor_model = SensorModel(self.batch_size, self.num_ensemble, self.dim_z)
        
    def call(self, inputs):

        raw_sensor = inputs

        # get sensor reading
        z = self.sensor_model(raw_sensor)
        z = tf.reshape(z, [self.batch_size, 1, self.dim_z])

        # expand z to ensembles
        for i in range (self.batch_size):
            if i == 0:
                ensemble_z = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
            else:
                tmp = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
                ensemble_z = tf.concat([ensemble_z, tmp], 0)
        # make sure the ensemble shape matches
        ensemble_z = tf.reshape(ensemble_z, [self.batch_size, self.num_ensemble, self.dim_z])

        # tuple structure of updated state
        output = (ensemble_z, z)

        return output

# Xiao's version - enKFMLP in high dimensional state space
class enKFMLP(tf.keras.Model):
    def __init__(self, batch_size, num_ensemble, **kwargs):
        super(enKFMLP, self).__init__()

        # initialization
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        
        self.dim_x = 200
        self.dim_z = 200

        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.1
        # self.r_diag = np.array([1, 0.01]).astype(np.float32)
        self.r_diag = self.r_diag.astype(np.float32)

        # learned process model
        self.process_model = BayesianProcessModel(self.batch_size, self.num_ensemble, self.dim_x)

        # learned observation model
        self.observation_model = ObservationModel(self.batch_size, self.num_ensemble, self.dim_x, self.dim_z)

        # learned observation noise
        self.observation_noise_model = ObservationNoise(self.batch_size, self.num_ensemble, self.dim_z, self.r_diag)

        # learned sensor model
        self.sensor_model = SensorModel(self.batch_size, self.num_ensemble, self.dim_z)

        # decoder model
        self.decoder = StateDecoderModel(self.batch_size, self.num_ensemble, self.dim_x)

        self.utils_ = utils()

        self.transform_ = transform()

    def call(self, inputs, states):
        # decompose inputs and states
        raw_sensor = inputs

        state_old, m_state = states

        state_old = tf.reshape(state_old, [self.batch_size, self.num_ensemble, self.dim_x])

        m_state = tf.reshape(m_state, [self.batch_size, self.dim_x])


        # get prediction and noise of next state
        state_p = self.process_model(state_old)
        state_pred = state_p

        # update step
        # get predicted observations
        H_X = self.observation_model(state_pred)
        # H_X = state_pred

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
        z = self.sensor_model(raw_sensor)

        # expand z to ensembles
        for i in range (self.batch_size):
            if i == 0:
                ensemble_z = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
            else:
                tmp = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
                ensemble_z = tf.concat([ensemble_z, tmp], 0)
        # make sure the ensemble shape matches
        ensemble_z = tf.reshape(ensemble_z, [self.batch_size, self.num_ensemble, self.dim_z])

        # get observation noise
        R, diag_R = self.observation_noise_model(z)

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
        state_new = state_pred + tf.transpose(tf.matmul(K, y_bar), perm=[0,2,1])

        # the ensemble state mean
        m_state_new = tf.reduce_mean(state_new, axis = 1)
        m_state_new = tf.reshape(m_state_new, [self.batch_size, 1, self.dim_x])

        true_state = self.decoder(state_new)
        m_true_state = tf.reduce_mean(true_state, axis = 1)
        m_true_state = tf.reshape(m_true_state, [self.batch_size, 1, 5])

        # tuple structure of updated state
        output = (state_new, m_state_new, true_state, m_true_state)

        return output