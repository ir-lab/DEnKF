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
from numpy.random import random

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class ProcessModel(tf.keras.Model):
    '''
    process model is taking the state and get a distribution of a prediction state,
    which is represented as ensemble.
    new_state = [batch_size, num_particles, dim_x]
    state vector dim_x -> fc 32 -> fc 64 -> fc 32 -> dim_x
    '''
    def __init__(self, batch_size, num_particles, dim_x):
        super(ProcessModel, self).__init__()
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.dim_x = dim_x

    def build(self, input_shape):
        self.process_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc1')
        self.process_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc2')
        self.process_fc3 = tf.keras.layers.Dense(
            units=2,
            activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc3')

    def call(self, last_state):
        last_state = tf.reshape(last_state, [self.batch_size * self.num_particles, self.dim_x])
        theta = tf.reshape(last_state[:,2], [self.batch_size * self.num_particles, 1])
        v = tf.reshape(last_state[:,3], [self.batch_size * self.num_particles, 1])
        theta_dot = tf.reshape(last_state[:,4], [self.batch_size * self.num_particles, 1])
        theta = theta + theta_dot
        st = tf.sin(theta)
        ct = tf.cos(theta)
        x = tf.reshape(last_state[:,0], [self.batch_size * self.num_particles, 1])
        y = tf.reshape(last_state[:,1], [self.batch_size * self.num_particles, 1])
        x = x + v*st
        y = y + v*ct
        data_in = tf.concat([v, theta_dot], axis = -1)
        fc1 = self.process_fc1(data_in)
        fc2 = self.process_fc2(fc1)
        update = self.process_fc3(fc2)
        data_out = data_in + update
        new_state = tf.concat([x, y, theta, data_out], axis = -1)
        new_state = tf.reshape(new_state, [self.batch_size, self.num_particles, self.dim_x])

        return new_state

class ProcessNoise(tf.keras.Model):
    '''
    Process noise model is used to learn the epistemic uncertainty of the process model,
    it models the diag(Q) of the covariance Q, Q varied at every timestep given different state.
    Q = [batch_size, dim_x, dim_x]
    i.e., 
    if the state has 6 inputs
    state vector 6 -> fc 32 -> fc 64 -> 6
    the result is the diag of Q where Q is a 6x6 matrix
    '''
    def __init__(self, batch_size, num_particles, dim_x, q_diag):
        super(ProcessNoise, self).__init__()
        self.batch_size = batch_size
        self.num_particles = num_particles
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

    def call(self, state):
        state = tf.reshape(state, [self.batch_size, self.dim_x])
        fc1 = self.process_noise_fc1(state)
        fc2 = self.process_noise_fc2(fc1)
        diag = self.process_noise_fc3(fc2)
        diag = tf.square(diag + self.learned_process_noise_bias)
        diag = diag + self.fixed_process_noise_bias
        mean = np.zeros((self.dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * self.batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = nd.sample(self.num_particles)
        Q = tf.reshape(nd.sample(self.num_particles), [self.batch_size, self.num_particles, self.dim_x])

        return Q

class BayesianImageSensorModelonly(tf.keras.Model):
    '''
    sensor model is used for learning a representation of the current observation,
    the representation can be explainable or latent.  
    observation = [batch_size, img_h, img_w, channel]
    encoding = [batch_size, dim_fc3]
    # 128, 64, 64, 32, 32, 32, fc 128, 64, 32, 32, dim_z
    '''
    def __init__(self, batch_size, num_ensemble, dim_z):
        super(BayesianImageSensorModelonly, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.num_ensemble = num_ensemble

    def build(self, input_shape):
        self.sensor_conv1 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=7,
            strides=[3, 3],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv1')
        self.sensor_conv2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=5,
            strides=[1, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv2')
        self.sensor_conv3 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3,
            strides=[1, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv3')
        self.sensor_conv4 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3,
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
        self.bayes_sensor_fc4 = tfp.layers.DenseFlipout(
            units=128,
            activation=tf.nn.relu,
            name='bayes_sensor_fc4')
        # self.bayes_sensor_fc5 = tfp.layers.DenseFlipout(
        #     units=self.dim_z,
        #     activation=None,
        #     name='bayes_sensor_fc5')

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
        
        inputs = self.flatten(conv5)
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

        fc1 = self.sensor_fc1(inputs_z)
        fc2 = self.sensor_fc2(fc1)
        fc4 = self.bayes_sensor_fc4(fc2)
        # observation = self.bayes_sensor_fc5(fc4)
        encoding = fc4

        # observation = tf.reshape(observation, [self.batch_size, self.num_ensemble, self.dim_z])
        # observation_m = tf.reduce_mean(observation, axis = 1)

        encoding = tf.reshape(encoding, [self.batch_size, self.num_ensemble, 128])
        encoding = tf.reduce_mean(encoding, axis = 1)

        return encoding

class Likelihood(tf.keras.Model):
    '''
    likelihood function is used to generate the probability for each particle with given
    observation encoding
    particles = [batch_size, num_particles, dim_x]
    like = [batch_size, num_particles]
    '''
    def __init__(self, batch_size, num_particles):
        super(Likelihood, self).__init__()
        self.batch_size = batch_size
        self.num_particles = num_particles

    def build(self, input_shape):
        self._fc_layer_1 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='_fc_layer_1')
        self._fc_layer_2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='_fc_layer_2')
        self._fc_layer_3 = tf.keras.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='_fc_layer_3')

    def call(self, inputs):
        # unpack the inputs
        encoding = inputs

        # expand the encoding into particles
        for n in range (self.batch_size):
            if n == 0:
                encodings = tf.reshape(tf.stack([encoding[n]] * self.num_particles), [1, self.num_particles, 128])
            else:
                tmp = tf.reshape(tf.stack([encoding[n]] * self.num_particles), [1, self.num_particles, 128])
                encodings = tf.concat([encodings, tmp], 0)

        encodings = tf.reshape(encodings, [self.batch_size * self.num_particles, 128])
        like = self._fc_layer_1(encodings)
        like = self._fc_layer_2(like)
        like = self._fc_layer_3(like)
        like = tf.reshape(like, [self.batch_size, self.num_particles])
        w = tf.reduce_sum(like, axis=1)
        w = tf.stack([w]*self.num_particles)
        w = tf.transpose(w, perm=[1,0])
        like = like/w
        return like

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

class transition_model(tf.keras.Model):
    def __init__(self, batch_size, num_particles,**kwargs):

        super(transition_model, self).__init__(**kwargs)

        # initialization
        self.batch_size = batch_size
        self.num_particles = num_particles
        
        self.dim_x = 5

        # learned process model
        self.process_model = ProcessModel(self.batch_size, self.num_particles, self.dim_x)

        self.transform_ = transform()

    def call(self, states):

        particles, weights, m_state = states

        # prediction step
        in_state = self.transform_.state_inv_transform(particles)
        state_p = self.process_model(in_state)
        state_p = self.transform_.state_transform(state_p)

        particles_new = state_p

        # use the new weights to calculate state
        weights = tf.expand_dims(weights,-1)
        tmp = tf.multiply(particles_new,weights)
        m_state = tf.reduce_sum(tmp, axis=1)
        m_state = tf.reshape(m_state, [self.batch_size, 1, self.dim_x])
        weights = tf.reshape(weights,[self.batch_size, self.num_particles])

        # tuple structure of updated state
        output =(particles_new, weights, m_state)

        return output

class Particle_filter(tf.keras.Model):
    def __init__(self, batch_size, num_particles, **kwargs):
        super(Particle_filter, self).__init__()

        # initialization
        self.batch_size = batch_size
        self.num_particles = num_particles
        
        self.dim_x = 5
        self.dim_z = 2

        self.q_diag = np.array([0.1,0.1,0.1,0.1,0.1]).astype(np.float32)
        self.q_diag = self.q_diag.astype(np.float32)

        # learned process model
        self.process_model = ProcessModel(self.batch_size, self.num_particles, self.dim_x)

        # learned process noise model
        self.process_noise_model = ProcessNoise(self.batch_size, self.num_particles, self.dim_x, self.q_diag)

        # learned likelihood model
        self.likelihood_model = Likelihood(self.batch_size, self.num_particles)

        # learned sensor model
        self.sensor_model = BayesianImageSensorModelonly(self.batch_size, 32, self.dim_z)

        self.utils_ = utils()

        self.transform_ = transform()

        self.alpha = 0.05

    def _resample(self, particles, weights):
        """
        Resample the particles to discard particles with low weights
        Parameters
        ----------
        particles : tensor [batch_size, num_particles, dim_x]
            old particle set
        weights : tensor [batch_size, num_particles]
            their weights
        training : bool
            training or testing?
        Returns
        -------
        new_particles: tensor [batch_size, num_particles, dim_x]
            resampled particle set
        new_weights : tensor [batch_size, num_particles]
            their weights
        """
        # weights are in log scale, to turn them into a distribution, we
        # exponentiate and normalize them == apply the softmax transform
        # weights = tf.nn.softmax(weights, axis=-1)

        # soft resampling - this maintains a gradient between old and new
        # weights
        resample_prob = (1 - self.alpha) * weights + \
            self.alpha/float(self.num_particles)
        new_weights = weights / resample_prob

        # systematic resampling: the samples are evenly distributed over the
        # original particles
        base_inds = \
            tf.linspace(0.0, (self.num_particles-1.)/float(self.num_particles),
                        self.num_particles)
        random_offsets = tf.random.uniform([self.batch_size], 0.0,
                                           1.0 / float(self.num_particles))
        # shape: batch_size x num_resampled
        inds = random_offsets[:, None] + base_inds[None, :]
        cum_probs = tf.cumsum(resample_prob, axis=1)

        # shape: batch_size x num_resampled x num_particles
        inds_matching = inds[:, :, None] < cum_probs[:, None, :]
        samples = tf.cast(tf.argmax(tf.cast(inds_matching, 'int32'),
                                    axis=2), 'int32')

        # compute 1D indices into the 2D array
        idx = samples + self.num_particles * tf.tile(
            tf.reshape(tf.range(self.batch_size), [self.batch_size, 1]),
            [1, self.num_particles])

        # index using the 1D indices and reshape again
        new_particles = \
            tf.gather(tf.reshape(particles,
                                 [self.batch_size*self.num_particles,
                                  self.dim_x]), idx)
        new_particles = \
            tf.reshape(new_particles,
                       [self.batch_size, self.num_particles, self.dim_x])

        new_weights = tf.gather(tf.reshape(new_weights,
                                           [self.batch_size*self.num_particles,
                                            1]), idx)

        new_weights = tf.reshape(new_weights,
                                 [self.batch_size, self.num_particles])
        # renormalize
        new_weights /= tf.reduce_sum(new_weights, axis=-1, keepdims=True)

        # # return into log scale
        # new_weights = tf.math.log(new_weights)

        return new_particles, new_weights

    def call(self, inputs, states):
        raw_sensor = inputs

        particles, weights, m_state = states

        # prediction step
        in_state = self.transform_.state_inv_transform(particles)
        state_p = self.process_model(in_state)
        state_p = self.transform_.state_transform(state_p)

        update = state_p - particles
        d_state = tf.reduce_mean(update, axis = 1)
        d_state = tf.reshape(d_state, [self.batch_size, 1, self.dim_x])
        particles_new = state_p

        m_state = tf.reshape(m_state, [self.batch_size, self.dim_x])

        Q = self.process_noise_model(m_state)

        particles_new = particles_new + Q
        
        # update step
        encoding = self.sensor_model(raw_sensor)

        like = self.likelihood_model(encoding)

        weights = weights + like

        # calculate new weights
        w = tf.reduce_sum(weights, axis=1)
        w = tf.stack([w]*self.num_particles)
        w = tf.transpose(w, perm=[1,0])
        weights = weights/w

        particles_new, new_weights = self._resample(particles_new, weights)

        # use the new weights to calculate state
        weights = tf.expand_dims(weights,-1)
        tmp = tf.multiply(particles_new,weights)
        m_state = tf.reduce_sum(tmp, axis=1)
        m_state = tf.reshape(m_state, [self.batch_size, 1, self.dim_x])
        weights = tf.reshape(weights,[self.batch_size, self.num_particles])

        out = (particles_new, weights, m_state, d_state)
        
        return out



        






