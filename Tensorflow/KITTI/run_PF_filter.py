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

import diff_PF
import diff_enKF
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
        batch_size = 4
        window_size = 8
        num_particles = 100

        # load the model
        model = diff_PF.Particle_filter(batch_size, num_particles)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # # load pre-trained weights
        gt_pre, gt_now, obs, raw_sensor, d_state = DataLoader.load_training_data(batch_size, add_noise=False, norm=True)
        init_states =  DataLoader.format_particle_state(gt_pre, batch_size, num_particles, dim_x)
        _ = model(raw_sensor,init_states)
        model.load_weights('./models/dPF_dPF_v1.2_KITTI001.h5')

        # # load a trained model
        # sensor_model = diff_enKF.StandaloneModel(batch_size, 32)
        # gt_pre, gt_now, obs, raw_sensor, d_state = DataLoader.load_training_data(batch_size, add_noise=False, norm=True)
        # init_states = DataLoader.format_particle_state(gt_pre, batch_size, num_particles, dim_x)
        # _ = model(raw_sensor,init_states)
        # _ = sensor_model(raw_sensor)
        # sensor_model.load_weights('./models/DEnKF_vS.08_sensor034.h5')
        # i = 0
        # for layer in model.layers[3].layers:
        #     layer.set_weights(sensor_model.layers[0].layers[i].get_weights())
        #     layer.trainable = False
        #     i = i +1

        epoch = 50
        counter = 0
        for k in range (epoch):
            print('end-to-end wholemodel')
            print("========================================= working on epoch %d =========================================: " % (k))
            # 
            steps = int(10000/batch_size)
            for step in range(steps):
                counter = counter + 1
                gt_pre, gt_now, obs, raw_sensor, d_state = DataLoader.load_training_data_seq(batch_size, window_size, norm=True)
                with tf.GradientTape(persistent=True) as tape:
                    loss = 0
                    loss_1 = 0
                    #### apply multiple steps for EKF #### 
                    for i in range (window_size):
                        start = time.time()
                        if i == 0:
                            states = DataLoader.format_particle_state(gt_pre[:,i,:,:], batch_size, num_particles, dim_x)
                            out = model(raw_sensor[:,i,:,:],states)
                        else:
                            out = model(raw_sensor[:,i,:,:],states)
                        state_h = out[2] # state output
                        state_d = out[-1] # transition residual output
                        loss_1 = loss_1 + get_loss._mse(d_state[:,i,:,:] - state_d) # state transition
                        loss = get_loss._mse(gt_now[:,i,:,:] - state_h) # end-to-end state
                        states = (out[0], out[1], out[2]) # update state
                        end = time.time()
                    if step % 50 ==0:
                        print("Training loss at step %d: %.4f (took %.3f seconds) " %
                                (step, float(loss), float(end-start)))
                        print(loss_1)
                        with train_summary_writer.as_default():
                            tf.summary.scalar('total_loss', loss, step=counter)
                            tf.summary.scalar('process_loss', loss_1, step=counter)
                        print('---')
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                grads = tape.gradient(loss_1, model.layers[0].trainable_weights)
                optimizer.apply_gradients(zip(grads, model.layers[0].trainable_weights))

            if (k+1) % epoch == 0:
                model.save_weights('./models/dPF_'+version+'_'+name[index]+str(epoch).zfill(3)+'.h5')
                print('model is saved at this epoch')
            if (k+1) % 2 ==0:
                model.save_weights('./models/dPF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                print('model is saved at this epoch')

                # define batch_size
                test_batch_size = 1

                test_num_particles = 100

                # load the model
                model_test = diff_PF.Particle_filter(test_batch_size, test_num_particles)
                test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(0, add_noise=False, norm=True)

                dataset = pickle.load(open('KITTI_VO_test.pkl', 'rb'))
                N = len(dataset)

                # load init state
                inputs = test_raw_sensor
                init_states = DataLoader.format_particle_state(test_gt_pre, test_batch_size, test_num_particles, dim_x)

                dummy = model_test(inputs, init_states)
                model_test.load_weights('./models/dPF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                for layer in model_test.layers:
                    layer.trainable = False
                model_test.summary()

                '''
                run a test demo and save the state of the test demo
                '''
                data = {}
                data_save = []
                gt_save = []

                for t in range (N):
                    if t == 0:
                        states = init_states
                    test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(t, add_noise=False, norm=True)
                    raw_sensor = test_raw_sensor
                    out = model_test(raw_sensor, states)
                    if t%10 == 0:
                        print('---')
                        print(out[0])
                        print(test_gt_now)
                    states = (out[0], out[1], out[2]) # update state
                    state_out = np.array(out[2])
                    gt_out = np.array(test_gt_now)
                    data_save.append(state_out)
                    gt_save.append(gt_out)
                data['state'] = data_save
                data['gt'] = gt_save
                with open('./output/dPF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'wb') as f:
                    pickle.dump(data, f)

    else:
        k_list = [1]
        for k in k_list:
            # define batch_size
            test_batch_size = 1

            test_num_particles = 100

            # load the model
            model_test = diff_PF.Particle_filter(test_batch_size, test_num_particles)
            test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(0, add_noise=False, norm=True, black = False)

            dataset = pickle.load(open('KITTI_VO_test.pkl', 'rb'))
            N = len(dataset)

            # load init state
            inputs = test_raw_sensor
            init_states = DataLoader.format_particle_state(test_gt_pre, test_batch_size, test_num_particles, dim_x)

            dummy = model_test(inputs, init_states)
            model_test.load_weights('./models/dPF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
            for layer in model_test.layers:
                layer.trainable = False
            model_test.summary()

            # load transition model
            transition_model = diff_PF.transition_model(test_batch_size, test_num_particles)
            _ = transition_model(init_states)
            transition_model.layers[0].set_weights(model_test.layers[0].get_weights())
            for layer in transition_model.layers:
                layer.trainable = False

            dataset = pickle.load(open('KITTI_VO_test.pkl', 'rb'))
            N = len(dataset)

            for j in range (2):
                '''
                run a test demo and save the state of the test demo
                '''
                data = {}
                data_save = []
                gt_save = []

                N = random.randint(0,700)
                use_transition = False
                for t in range (N, N+800):
                    test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(t, add_noise=False, norm=True, black =False)
                    if t == N+1 or t == N+2:
                        test_gt_pre, test_gt_now, test_obs, test_raw_sensor, _ = DataLoader.load_testing_data_onebyone(t, add_noise=False, norm=True, black =True)
                    if t == N:
                        states = DataLoader.format_particle_state(test_gt_pre, test_batch_size, test_num_particles, dim_x)
                    elif t % 50 == 0:
                        states = DataLoader.format_particle_state(test_gt_pre, test_batch_size, test_num_particles, dim_x)
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
                        print(out[2])
                        print(test_gt_now)
                    states = (out[0], out[1], out[2]) # update state
                    state_out = np.array(out[2])
                    gt_out = np.array(test_gt_now)
                    data_save.append(state_out)
                    gt_save.append(gt_out)
                data['state'] = data_save
                data['gt'] = gt_save
                with open('./output/dPF_'+version+'_'+ name[index]+str(k).zfill(3)+'test'+str(j)+'_black.pkl', 'wb') as f:
                    pickle.dump(data, f)
        
'''
load loss functions
'''
get_loss = diff_PF.getloss()

'''
load data for training
'''
global name 
name = ['KITTI']

global index
index = 0

global version
version = 'dPF_v1.3'
old_version = version

# os.system('sudo rm -rf /tf/experiments/loss/'+version+'/')

# train_log_dir = "/tf/experiments/loss/"+version
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)

def main():
    # training = True
    # run_filter(training)

    training = False
    run_filter(training)

if __name__ == "__main__":
    main()