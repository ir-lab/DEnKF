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

import diff_enKF_high as diff_enKF
from dataloader_v2 import DataLoader
DataLoader = DataLoader()

'''
define the training loop
'''
def run_filter(mode):

    tf.keras.backend.clear_session()
    dim_x = 200
    if mode == True:
        # define batch_size
        batch_size = 16

        # define number of ensemble
        num_ensemble = 32

        # load the model
        model = diff_enKF.enKFMLP(batch_size, num_ensemble)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        sensor_model = diff_enKF.StandaloneModel(batch_size, num_ensemble)

        # initialize the model
        gt_pre, gt_now, obs, raw_sensor, d_state = DataLoader.load_training_data(batch_size, add_noise=False, norm=True)
        init_states = sensor_model(raw_sensor)
        _ = model(raw_sensor,init_states)

        # # load a trained model
        # sensor_model = diff_enKF.StandaloneModel(batch_size, num_ensemble)
        # gt_pre, gt_now, obs, raw_sensor, d_state = DataLoader.load_training_data(batch_size, add_noise=False, norm=True)
        # init_states = DataLoader.format_state(gt_pre, batch_size, num_ensemble, dim_x)
        # _ = model(raw_sensor,init_states)
        # _ = sensor_model(raw_sensor)
        # sensor_model.load_weights('./models/DEnKF_vS.08_sensor034.h5')
        # model.layers[3].set_weights(sensor_model.layers[0].get_weights())
        # model.layers[3].trainable = False

        epoch = 50
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
                    sensor_model.layers[0].set_weights(model.layers[3].get_weights())
                    states = sensor_model(raw_sensor)
                    out = model(raw_sensor,states)
                    state_h = out[-1]
                    loss = get_loss._mse(gt_now - state_h) # end-to-end state
                    end = time.time()
                    if step % 50 ==0:
                        print("Training loss at step %d: %.4f (took %.3f seconds) " %
                              (step, float(loss), float(end-start)))
                        with train_summary_writer.as_default():
                            tf.summary.scalar('total_loss', loss, step=counter)
                        print('---')
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                # grads = tape.gradient(loss_1, model.layers[0].trainable_weights)
                # optimizer.apply_gradients(zip(grads, model.layers[0].trainable_weights))
                # grads = tape.gradient(loss_2, model.layers[3].trainable_weights)
                # optimizer_sensor.apply_gradients(zip(grads, model.layers[3].trainable_weights))
                # grads = tape.gradient(loss_3, model.layers[1].trainable_weights)
                # optimizer.apply_gradients(zip(grads, model.layers[1].trainable_weights))

            if (k+1) % epoch == 0:
                model.save_weights('./models/DEnKF_'+version+'_'+name[index]+str(epoch).zfill(3)+'.h5')
                print('model is saved at this epoch')
            if (k+1) % 2 ==0:
                model.save_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                print('model is saved at this epoch')

                # define batch_size
                test_batch_size = 1

                test_num_ensemble = 32

                # load the model
                model_test = diff_enKF.enKFMLP(test_batch_size, test_num_ensemble)
                sensor_model_test = diff_enKF.StandaloneModel(test_batch_size, test_num_ensemble)
                test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(0, add_noise=False, norm=True)

                dataset = pickle.load(open('KITTI_VO_test.pkl', 'rb'))
                N = len(dataset)

                # load init state
                inputs = test_raw_sensor
                init_states = sensor_model_test(inputs)
                _ = model_test(inputs, init_states)

                model_test.load_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                for layer in model_test.layers:
                    layer.trainable = False
                sensor_model_test.layers[0].set_weights(model_test.layers[3].get_weights())
                model_test.summary()

                '''
                run a test demo and save the state of the test demo
                '''
                data = {}
                data_save = []
                gt_save = []
                for t in range (N):
                    test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(t, add_noise=False, norm=True)
                    raw_sensor = test_raw_sensor
                    if t == 0:
                        states = sensor_model_test(raw_sensor)
                    out = model_test(raw_sensor, states)
                    if t%10 == 0:
                        print('---')
                        print(out[3])
                        print(test_gt_now)
                    states = (out[0], out[1])
                    state_out = np.array(out[3])
                    gt_out = np.array(test_gt_now)
                    data_save.append(state_out)
                    gt_save.append(gt_out)
                data['state'] = data_save
                data['gt'] = gt_save
                with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'wb') as f:
                    pickle.dump(data, f)

    else:
        k_list = [31]
        for k in k_list:

            # define batch_size
            test_batch_size = 1

            test_num_ensemble = 32

            # load the model
            model_test = diff_enKF.enKFMLP_v2(test_batch_size, test_num_ensemble)
            test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(0, add_noise=False, norm=True)

            # load init state
            inputs = test_raw_sensor
            init_states = DataLoader.format_init_state(test_gt_pre, test_batch_size, test_num_ensemble, dim_x)

            dummy = model_test(inputs, init_states)
            model_test.load_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
            for layer in model_test.layers:
                layer.trainable = False
            model_test.summary()

            # load the transition model
            transition_model = diff_enKF.bayesiantransition(test_batch_size, test_num_ensemble)
            _ = transition_model(init_states)
            transition_model.layers[0].set_weights(model_test.layers[0].get_weights())
            for layer in transition_model.layers:
                layer.trainable = False

            dataset = pickle.load(open('KITTI_VO_test.pkl', 'rb'))
            N = len(dataset)

            # '''
            # run a test demo and save the state of the test demo
            # '''
            # for j in range (2):
            #     data = {}
            #     data_save = []
            #     emsemble_save = []
            #     gt_save = []
            #     transition_save = []
            #     observation_save = []

            #     N = random.randint(0,700)
            #     use_transition = True

            #         # if use_transition == True
            #         #     draw = random.uniform(0, 1)
            #         #     if draw > 0.3:
            #         #         transition_model(state)
            #         #     else:

            #     for t in range (N, N+800):
            #         test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(t, add_noise=False, norm=True)
            #         if t == 0:
            #             states = DataLoader.format_init_state(test_gt_pre, test_batch_size, test_num_ensemble, dim_x)
            #         raw_sensor = test_raw_sensor
            #         out = model_test(raw_sensor, states)
            #         if t%50 == 0:
            #             print('---')
            #             print(out[1])
            #             print(test_gt_now)
            #         # if t%2 == 0:
            #         #     print('---')
            #         #     print('transition:')
            #         #     print(out[2])
            #         #     print('after H:')
            #         #     print(out[5])
            #         #     print('sensor input:')
            #         #     print(out[3])
            #         #     print('sensor gt:')
            #         #     print(test_obs)
            #         #     print('output:')
            #         #     print(out[1])
            #         #     print('gt:')
            #         #     print(test_gt_now)
            #         states = (out[0], out[1])
            #         state_out = np.array(out[1])
            #         gt_out = np.array(test_gt_now)
            #         ensemble = np.array(tf.reshape(out[0], [test_num_ensemble, dim_x]))
            #         transition_out = np.array(out[2])
            #         observation_out = np.array(out[3])
            #         data_save.append(state_out)
            #         emsemble_save.append(ensemble)
            #         gt_save.append(gt_out)
            #         observation_save.append(observation_out)
            #         transition_save.append(transition_out)
            #     data['state'] = data_save
            #     data['ensemble'] = emsemble_save
            #     data['gt'] = gt_save
            #     data['observation'] = observation_save
            #     data['transition'] = transition_save

            #     with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'test'+str(j)+'_long.pkl', 'wb') as f:
            #         pickle.dump(data, f)

            '''
            run a test demo with missing observation
            '''
            for j in range (2):
                data = {}
                data_save = []
                gt_save = []

                N = random.randint(0,700)
                use_transition = True

                for t in range (N, N+800):
                    test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(t, add_noise=False, norm=True)
                    if t == N:
                        states = DataLoader.format_init_state(test_gt_pre, test_batch_size, test_num_ensemble, dim_x)
                    raw_sensor = test_raw_sensor
                    if use_transition == True:
                        draw = random.uniform(0, 1)
                        if draw < 0.3:
                            out = transition_model(states)
                        else:
                            out = model_test(raw_sensor, states)
                    else:
                        out = model_test(raw_sensor, states)
                    if t%50 == 0:
                        print('---')
                        print(out[1])
                        print(test_gt_now)
                    states = (out[0], out[1])
                    state_out = np.array(out[1])
                    gt_out = np.array(test_gt_now)
                    data_save.append(state_out)
                    gt_save.append(gt_out)
                data['state'] = data_save
                data['gt'] = gt_save

                with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'test'+str(j)+'_missing.pkl', 'wb') as f:
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
version = 'high_v1.0'
old_version = version

os.system('rm -rf /tf/experiments/loss/high_v1.0')

train_log_dir = "/tf/experiments/loss/high_v1.0"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

def main():
    training = True
    run_filter(training)

    # training = False
    # run_filter(training)

if __name__ == "__main__":
    main()