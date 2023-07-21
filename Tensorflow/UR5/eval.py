import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import numpy as np 
import pickle
import pdb
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

global name 
name = ['joint', 'EE', 'all']
global index
index = 0

global version
version = 'v8.0-ur5'

# version = 'vbohg-img'
# version = 'vhalf-img'


ind = 0
scale = 85.
num_points = 200
# 182

'''
visualize the states
'''
dim_x = 7
k_list = [119]
for k in k_list:
    plt_observation = []
    plt_pred = []
    gt_state = []
    ori_gt = []
    ori_pred = []
    with open('./output/bayes_enkf_'+version+'_'+ name[index]+str(k).zfill(3)+'test.pkl', 'rb') as f:
        data = pickle.load(f)
        test_demo = data['state']
        ensemble = data['ensemble']
        gt_data = data['gt']
        plt_observation = data['transition']

    gt_state = np.array(gt_data)
    plt_pred = np.array(test_demo)
    ensemble = np.array(ensemble)
    gt_state = gt_state[0:num_points].reshape((num_points, dim_x))
    plt_pred = plt_pred[0:num_points].reshape((num_points, dim_x))
    gt_state = gt_state[0:num_points].reshape((num_points, dim_x))

    plt_observation = np.array(plt_observation)
    plt_observation = plt_observation[0:num_points].reshape((num_points, dim_x))


    uncertain = np.array(ensemble)
    uncertain = uncertain[0:num_points]
    en_max = np.amax(uncertain, axis = 1)
    en_min = np.amin(uncertain, axis = 1)

# idx = 9
# rmse1 = mean_squared_error(plt_pred[:,idx], gt_state[:,idx], squared=False)
# rmse2 = mean_squared_error(plt_observation[:,idx], gt_state[:,idx], squared=False)
# print('rmse: final, transition')
# print(rmse1, rmse2)

# mae1 = mean_squared_error(plt_pred[:,idx], gt_state[:,idx])
# mae2 = mean_squared_error(plt_observation[:,idx], gt_state[:,idx])
# print('mae: final, transition')
# print(mae1, mae2)

rmse1 = mean_squared_error(plt_pred, gt_state, squared=False)
rmse2 = mean_squared_error(plt_observation, gt_state, squared=False)
print('rmse: final, transition')
print(rmse1, rmse2)

mae1 = mean_squared_error(plt_pred, gt_state)
mae2 = mean_squared_error(plt_observation, gt_state)
print('mae: final, transition')
print(mae1, mae2)




# index = 0
# version = 'v8.0-ur5'
# dim_x = 7
# num_points = 200

# k_list = [99]
# for k in k_list:
#     plt_observation = []
#     plt_pred = []
#     gt_state = []
#     ori_gt = []
#     ori_pred = []
#     with open('./output/bayes_enkf_'+version+'_'+ name[index]+str(k).zfill(3)+'test.pkl', 'rb') as f:
#         data = pickle.load(f)
#         test_demo = data['state']
#         ensemble = data['ensemble']
#         gt_data = data['gt']
#         plt_observation = data['transition']

#     gt_state = np.array(gt_data)
#     plt_pred = np.array(test_demo)
#     ensemble = np.array(ensemble)
#     gt_state = gt_state[0:num_points].reshape((num_points, dim_x))
#     plt_pred = plt_pred[0:num_points].reshape((num_points, dim_x))
#     gt_state = gt_state[0:num_points].reshape((num_points, dim_x))

#     plt_observation = np.array(plt_observation)
#     plt_observation = plt_observation[0:num_points].reshape((num_points, dim_x))

#     uncertain = np.array(ensemble)
#     uncertain = uncertain[0:num_points]
#     en_max = np.amax(uncertain, axis = 1)
#     en_min = np.amin(uncertain, axis = 1)

# # rmse1_prime = mean_squared_error(plt_pred[:,idx], gt_state[:,idx], squared=False)
# # rmse2_prime = mean_squared_error(plt_observation[:,idx], gt_state[:,idx], squared=False)
# # print('rmse: final, transition')
# # print(rmse1-rmse1_prime, rmse2-rmse2_prime)

# # mae1_prime = mean_squared_error(plt_pred[:,idx], gt_state[:,idx])
# # mae2_prime = mean_squared_error(plt_observation[:,idx], gt_state[:,idx])
# # print('mae: final, transition')
# # print(mae1-mae1_prime, mae2-mae2_prime)

# rmse1_prime = mean_squared_error(plt_pred, gt_state, squared=False)
# rmse2_prime = mean_squared_error(plt_observation, gt_state, squared=False)
# print('rmse: final, transition')
# print(rmse1-rmse1_prime, rmse2-rmse2_prime)

# mae1_prime = mean_squared_error(plt_pred, gt_state)
# mae2_prime = mean_squared_error(plt_observation, gt_state)
# print('mae: final, transition')
# print(mae1-mae1_prime, mae2-mae2_prime)

