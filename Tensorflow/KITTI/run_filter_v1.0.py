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

import diff_enKF
from dataloader_v2 import DataLoader
DataLoader = DataLoader()

'''
define the training loop
'''
def run_filter(mode):

    tf.keras.backend.clear_session()
    dim_x = 2
    if mode == True:
        # define batch_size
        batch_size = 16

        # define number of ensemble
        num_ensemble = 32

        # load the model
        model = diff_enKF.StandaloneModel(batch_size, num_ensemble)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # # load weights from a trained model
        # gt_pre, gt_now, obs, raw_sensor = DataLoader.load_training_data(batch_size,add_noise=False)
        # _ = model(raw_sensor)
        # model.load_weights('./models/DEnKF_v3.8_sensor019.h5')

        epoch = 100
        counter = 0
        for k in range (epoch):
            print('end-to-end wholemodel')
            print("========================================= working on epoch %d =========================================: " % (k))
            # 
            steps = int(21590/batch_size)
            for step in range(steps):
                counter = counter + 1
                gt_pre, gt_now, obs, raw_sensor, d_state = DataLoader.load_training_data(batch_size, add_noise=False, norm=True)
                with tf.GradientTape(persistent=True) as tape:
                    start = time.time()
                    out = model(raw_sensor)
                    y = out[1]
                    loss_1 = get_loss._mse(obs[:,:,0] - y[:,:,0])
                    loss_2 = get_loss._mse(obs[:,:,1] - y[:,:,1])
                    loss = 0.6 * loss_1 + 0.4 * loss_2 # sensor model
                    end = time.time()
                    if step % 50 ==0:
                        print("Training loss at step %d: %.4f (took %.3f seconds) " %
                              (step, float(loss), float(end-start)))
                        with train_summary_writer.as_default():
                            tf.summary.scalar('sensor_loss', loss, step=counter)
                            tf.summary.scalar('sensor_loss1', loss_1, step=counter)
                            tf.summary.scalar('sensor_loss2', loss_2, step=counter)
                        print(loss_1)
                        print(loss_2)
                        print('---')
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if (k+1) % epoch == 0:
                model.save_weights('./models/DEnKF_'+version+'_'+name[index]+str(epoch).zfill(3)+'.h5')
                print('model is saved at this epoch')
            if (k+1) % 5 ==0:
                model.save_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                print('model is saved at this epoch')

                # define batch_size
                test_batch_size = 1

                test_num_ensemble = 32

                # load the model
                model_test = diff_enKF.StandaloneModel(test_batch_size, test_num_ensemble)
                test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(0, add_noise=False, norm=True)

                # load init state
                inputs = test_raw_sensor

                dummy = model_test(inputs)
                model_test.load_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                for layer in model_test.layers:
                    layer.trainable = False
                model_test.summary()

                dataset = pickle.load(open('KITTI_VO_test.pkl', 'rb'))
                N = len(dataset)

                '''
                run a test demo and save the state of the test demo
                '''
                data = {}
                observation_save = []
                gt_observation = []

                for t in range (N):
                    test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(t, add_noise=False, norm=True)
                    raw_sensor = test_raw_sensor
                    out = model_test(raw_sensor)
                    if t%10 == 0:
                        print('---')
                        print(out[1])
                        print(test_obs)
                    gt_out = np.array(test_gt_now)
                    # ensemble = np.array(tf.reshape(out[0], [test_num_ensemble, dim_x]))

                    observation_out = np.array(out[1])
                    gt_observation_out = np.array(test_obs)

                    observation_save.append(observation_out)
                    gt_observation.append(gt_observation_out)

                data['observation'] = observation_save
                data['gt_observation'] = gt_observation

                with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'wb') as f:
                    pickle.dump(data, f)

    else:
        k_list = [99]
        for k in k_list:
            # define batch_size
            test_batch_size = 1

            test_num_ensemble = 32

            # load the model
            model_test = diff_enKF.StandaloneModel(test_batch_size, test_num_ensemble)
            test_gt_pre, test_gt_now, test_obs, test_raw_sensor = DataLoader.load_testing_data_onebyone(0, add_noise=False)

            # load init state
            inputs = test_raw_sensor

            dummy = model_test(inputs)
            model_test.load_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
            for layer in model_test.layers:
                layer.trainable = False
            model_test.summary()

            dataset = pickle.load(open('KITTI_VO_test.pkl', 'rb'))
            N = len(dataset)

            '''
            run a test demo and save the state of the test demo
            '''
            data = {}
            observation_save = []
            gt_observation = []

            for t in range (N):
                test_gt_pre, test_gt_now, test_obs, test_raw_sensor = DataLoader.load_testing_data_onebyone(t, add_noise=False)
                raw_sensor = test_raw_sensor
                out = model_test(raw_sensor)
                if t%10 == 0:
                    print('---')
                    print(out[1])
                    print(test_obs)
                gt_out = np.array(test_gt_now)
                # ensemble = np.array(tf.reshape(out[0], [test_num_ensemble, dim_x]))

                observation_out = np.array(out[1])
                gt_observation_out = np.array(test_obs)

                observation_save.append(observation_out)
                gt_observation.append(gt_observation_out)

            data['observation'] = observation_save
            data['gt_observation'] = gt_observation

            with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'test'+str(j)+'.pkl', 'wb') as f:
                pickle.dump(data, f)
        
'''
load loss functions
'''
get_loss = diff_enKF.getloss()

'''
load data for training
'''
global name 
name = ['KITTI', 'sensor']

global index
index = 1

global version
version = 'vS.08'
old_version = version

os.system('rm -rf /tf/experiments/loss/vS.08')

train_log_dir = "/tf/experiments/loss/vS.08"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

def main():
    training = True
    run_filter(training)

    # training = False
    # run_filter(training)

if __name__ == "__main__":
    main()