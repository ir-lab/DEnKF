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

import diff_enKF_new as diff_enKF
from dataloader_v2 import DataLoader
DataLoader = DataLoader()


'''
define the training loop
'''
def run_filter(mode):

    tf.keras.backend.clear_session()
    dim_x = 5
    if mode == True:
        # define batch_size
        batch_size = 32

        # define number of ensemble
        num_ensemble = 32

        # load the model
        model = diff_enKF.enKFMLP_v4(batch_size, num_ensemble)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        epoch = 100
        counter = 0

        for k in range (epoch):
            print('end-to-end wholemodel')
            print("========================================= working on epoch %d =========================================: " % (k))
            # 
            steps = int(21590/batch_size)
            for step in range(steps):
                counter = counter + 1
                gt_pre, gt_now, obs, raw_sensor, d_state = DataLoader.load_training_data(batch_size, add_noise=True, norm=True)
                with tf.GradientTape(persistent=True) as tape:
                    start = time.time()
                    states = DataLoader.format_state(gt_pre, batch_size, num_ensemble, dim_x)
                    out = model(raw_sensor, states)
                    state_h = out[1]
                    state_d = out[-1]
                    y = out[3]
                    m = out[5]
                    loss_1 = get_loss._mse(d_state - state_d) # state transition
                    # loss_2 = get_loss._mse(obs - y) # sensor model
                    loss_3 = get_loss._mse(obs - m) # observation model
                    loss = get_loss._mse(gt_now - state_h) # end-to-end state
                    end = time.time()
                    if step % 50 ==0:
                        print("Training loss at step %d: %.4f (took %.3f seconds) " %
                              (step, float(loss), float(end-start)))
                        print(loss_1)
                        print(loss_3)
                        print(state_h[0])
                        print(gt_now[0])
                        with train_summary_writer.as_default():
                            tf.summary.scalar('total_loss', loss, step=counter)
                            tf.summary.scalar('observation_loss', loss_3, step=counter)
                            tf.summary.scalar('process_loss', loss_1, step=counter)
                        print('---')
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                grads = tape.gradient(loss_1, model.layers[0].trainable_weights)
                optimizer.apply_gradients(zip(grads, model.layers[0].trainable_weights))

                grads = tape.gradient(loss_3, model.layers[1].trainable_weights)
                optimizer.apply_gradients(zip(grads, model.layers[1].trainable_weights))

            if (k+1) % epoch == 0:
                model.save_weights('./models/DEnKF_'+version+'_'+name[index]+str(epoch).zfill(3)+'.h5')
                print('model is saved at this epoch')
            if (k+1) % 10 ==0:
                model.save_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                print('model is saved at this epoch')

                # define batch_size
                test_batch_size = 1

                test_num_ensemble = 32

                # load the model
                model_test = diff_enKF.enKFMLP_v4(test_batch_size, test_num_ensemble)
                test_gt_pre, test_gt_now, test_obs, test_raw_sensor, d_state = DataLoader.load_testing_data(add_noise=True, norm=True)

                # load init state
                inputs = test_raw_sensor
                init_states = DataLoader.format_init_state(test_gt_pre, test_batch_size, test_num_ensemble, dim_x)

                dummy = model_test(inputs, init_states)
                model_test.load_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                for layer in model_test.layers:
                    layer.trainable = False
                model_test.summary()

                '''
                run a test demo and save the state of the test demo
                '''
                data = {}
                data_save = []
                emsemble_save = []
                gt_save = []
                transition_save = []
                observation_save = []

                for t in range (test_gt_now.shape[0]):
                    if t == 0:
                        states = init_states
                    raw_sensor = test_obs[t]
                    out = model_test(raw_sensor, states)
                    if t%10 == 0:
                        print('---')
                        print(out[1])
                        print(test_gt_now[t])
                    states = (out[0], out[1])
                    state_out = np.array(out[1])
                    gt_out = np.array(test_gt_now[t])
                    ensemble = np.array(tf.reshape(out[0], [test_num_ensemble, dim_x]))
                    transition_out = np.array(out[2])
                    observation_out = np.array(out[3])
                    data_save.append(state_out)
                    emsemble_save.append(ensemble)
                    gt_save.append(gt_out)
                    observation_save.append(observation_out)
                    transition_save.append(transition_out)
                data['state'] = data_save
                data['ensemble'] = emsemble_save
                data['gt'] = gt_save
                data['observation'] = observation_save
                data['transition'] = transition_save

                with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'wb') as f:
                    pickle.dump(data, f)
        
    else:
        k_list = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        for k in k_list:
            # define batch_size
            test_batch_size = 1

            test_num_ensemble = 32

            # load the model
            model_test = diff_enKF.enKFMLP_v4(test_batch_size, test_num_ensemble)
            test_gt_pre, test_gt_now, test_obs, test_raw_sensor, d_state = DataLoader.load_testing_data(add_noise=True, norm=True)

            # load init state
            inputs = test_obs[0]
            init_states = DataLoader.format_init_state(test_gt_pre[0], test_batch_size, test_num_ensemble, dim_x)

            dummy = model_test(inputs, init_states)
            model_test.load_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
            for layer in model_test.layers:
                layer.trainable = False
            model_test.summary()

            '''
            run a test demo and save the state of the test demo
            '''
            data = {}
            data_save = []
            emsemble_save = []
            gt_save = []
            transition_save = []
            observation_save = []

            for t in range (test_gt_now.shape[0]):
                if t == 0:
                    states = init_states
                raw_sensor = test_obs[t]
                out = model_test(raw_sensor, states)
                if t%50 == 0:
                    print('---------')
                    # print('transition:')
                    # print(out[2])
                    # print('after H:')
                    # print(out[5])
                    # print('sensor input:')
                    # print(raw_sensor)
                    print('output:')
                    print(out[1])
                    print('gt:')
                    print(test_gt_now[t])
                states = (out[0], out[1])
                state_out = np.array(out[1])
                gt_out = np.array(test_gt_now[t])
                ensemble = np.array(tf.reshape(out[0], [test_num_ensemble, dim_x]))
                transition_out = np.array(out[2])
                observation_out = np.array(out[3])
                data_save.append(state_out)
                emsemble_save.append(ensemble)
                gt_save.append(gt_out)
                observation_save.append(observation_out)
                transition_save.append(transition_out)
            data['state'] = data_save
            data['ensemble'] = emsemble_save
            data['gt'] = gt_save
            data['observation'] = observation_save
            data['transition'] = transition_save

            with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'test.pkl', 'wb') as f:
                pickle.dump(data, f)


'''
load loss functions
'''
get_loss = diff_enKF.getloss()

'''
load data for training
'''
global name 
name = ['KITTI']

global index
index = 0

global version
version = 'v4.92'
old_version = version

# os.system('rm -rf /tf/experiments/loss/v4.92')

# train_log_dir = "/tf/experiments/loss/v4.92"
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)

def main():
    # training = True
    # run_filter(training)

    training = False
    run_filter(training)

if __name__ == "__main__":
    main()