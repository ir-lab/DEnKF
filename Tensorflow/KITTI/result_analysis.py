import pickle
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['savefig.dpi'] = 500
import numpy as np
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import pickle
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

global name 
name = ['KITTI', 'sensor']

global index
index = 0

global filter_index
filter_index = 2

global version
version = 'dPF_v1.3'
old_version = version
# 'dEKF_v1.1'
# 'v5.09'
# 'dPF_v1.3'
# 'high_v1.0'
global filters
filters = ["DenKF", "dEKF", "dPF", "dPF_lrn"]

only_sensor = False
dim_x = 5

def motion_model(inputs):
	v, theta_dot = inputs
	x = 0
	y = 0
	theta = 0
	final_x = []
	final_y = []
	v.shape[0]
	# print(v.shape)
	# print(theta_dot.shape)
	for i in range (v.shape[0]):
		# dt = 0.103
		theta = theta + theta_dot[i]
		x = x + v[i]* np.sin(theta)
		y = y + v[i]* np.cos(theta)
		final_x.append(x)
		final_y.append(y)
	final_x = np.array(final_x)
	final_y = np.array(final_y)
	return final_x, final_y

def inv_transform(state):
	# parameters = pickle.load(open('parameters.pkl', 'rb'))
	parameters = pickle.load(open('full_parameters.pkl', 'rb'))
	obs_m = parameters['obs_m']
	obs_std = parameters['obs_std']
	state = state * obs_std + obs_m
	return state

def inv_transform_full(state):
	# parameters = pickle.load(open('parameters.pkl', 'rb'))
	parameters = pickle.load(open('full_parameters.pkl', 'rb'))
	state_m = parameters['state_m']
	state_std = parameters['state_std']
	state = state * state_std + state_m
	return state

def distance_metrics(gt, pred):
	result = []
	for step in [100, 200, 400, 800]:
		dist_gt = 0
		dist_pred = 0
		for i in range (step-1):
			x1 = gt[i,0]
			y1 = gt[i,1]
			x2 = gt[i+1,0]
			y2 = gt[i+1,1]
			dist_gt = dist_gt + np.linalg.norm(np.array([x1,y1])- np.array([x2,y2]))
			x1 = pred[i,0]
			y1 = pred[i,1]
			x2 = pred[i+1,0]
			y2 = pred[i+1,1]
			dist_pred = dist_pred + np.linalg.norm(np.array([x1,y1])- np.array([x2,y2]))
		result.append(abs(dist_pred-dist_gt)/dist_gt)

	result1 = result[0]
	result2 = np.mean(np.array(result))
	out = [result1, result2]
	return result1

def rotation_metrics(gt, pred):
	result = []
	for step in [100, 200, 400, 800]:
		dist_gt = 0
		for i in range (step-1):
			x1 = gt[i,0]
			y1 = gt[i,1]
			x2 = gt[i+1,0]
			y2 = gt[i+1,1]
			dist_gt = dist_gt + np.linalg.norm(np.array([x1,y1])- np.array([x2,y2]))
		theta_gt = gt[step-1,2]
		theta_pred = pred[step-1,2]
		err = abs(theta_pred-theta_gt)/dist_gt * 180 / np.pi
		result.append(err)
	result1 = result[0]
	result2 = np.mean(np.array(result))
	out = [result1, result2]
	return result1

def eval_metric(arr):
	return np.mean(arr), np.square(np.std(arr))

def evaluation():
	k = 1
	rmse_list = []
	mae_list = []
	trans = []
	rot = []
	for j in range (2):
		with open('./output/'+filters[filter_index]+'_'+version+'_'+ name[index]+str(k).zfill(3)+'test'+str(j)+'_black.pkl', 'rb') as f: #'+str(j)+'
			data = pickle.load(f)
			if only_sensor == True:
				gt_data = data['gt_observation']
				obs = data['observation']
			elif filters[filter_index] == "DenKF":
				test_demo = data['state']
				# ensemble = data['ensemble']
				gt_data = data['gt']
				# obs = data['observation']
			elif filters[filter_index] == "dEKF":
				test_demo = data['state']
				gt_data = data['gt']
			elif filters[filter_index] == "dPF":
				test_demo = data['state']
				gt_data = data['gt']
			elif filters[filter_index] == "dPF_lrn":
				test_demo = data['state']
				gt_data = data['gt']
		if filters[filter_index] == "DenKF":
			pred = np.array(test_demo)
			gt = np.array(gt_data)
			# obs = np.array(obs)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			# obs = np.reshape(obs, (obs.shape[0], 2))
			# ensemble = np.array(ensemble)
			# uncertain = np.array(ensemble)
			# en_max = np.amax(uncertain, axis = 1)
			# en_min = np.amin(uncertain, axis = 1)
			x = np.linspace(1, gt.shape[0], gt.shape[0])
			if dim_x == 5:
				# obs = inv_transform(obs)
				pred = inv_transform_full(pred)
				gt = inv_transform_full(gt)
				# sensor = (obs[:,0], obs[:,1])
				# obs_x, obs_y  = motion_model(sensor)
				trans_err = distance_metrics(gt, pred)
				rot_err = rotation_metrics(gt, pred)
				rmse = mean_squared_error(pred, gt, squared=False)
				mae = mean_absolute_error(pred, gt)
				rmse_list.append(rmse)
				mae_list.append(mae)
				trans.append(trans_err)
				rot.append(rot_err)
		elif filters[filter_index] == "dEKF":
			pred = np.array(test_demo)
			gt = np.array(gt_data)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			x = np.linspace(1, gt.shape[0], gt.shape[0])
			trans_err = distance_metrics(gt, pred)
			rot_err = rotation_metrics(gt, pred)
			rmse = mean_squared_error(pred, gt, squared=False)
			mae = mean_absolute_error(pred, gt)
			rmse_list.append(rmse)
			mae_list.append(mae)
			trans.append(trans_err)
			rot.append(rot_err)
		elif filters[filter_index] == "dPF":
			pred = np.array(test_demo)
			gt = np.array(gt_data)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			x = np.linspace(1, gt.shape[0], gt.shape[0])
			pred = inv_transform_full(pred)
			gt = inv_transform_full(gt)
			trans_err = distance_metrics(gt, pred)
			rot_err = rotation_metrics(gt, pred)
			rmse = mean_squared_error(pred, gt, squared=False)
			mae = mean_absolute_error(pred, gt)
			rmse_list.append(rmse)
			mae_list.append(mae)
			trans.append(trans_err)
			rot.append(rot_err)
		elif filters[filter_index] == "dPF_lrn":
			pred = np.array(test_demo)
			gt = np.array(gt_data)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			x = np.linspace(1, gt.shape[0], gt.shape[0])
			pred = inv_transform_full(pred)
			gt = inv_transform_full(gt)
			trans_err = distance_metrics(gt, pred)
			rot_err = rotation_metrics(gt, pred)
			rmse = mean_squared_error(pred, gt, squared=False)
			mae = mean_absolute_error(pred, gt)
			rmse_list.append(rmse)
			mae_list.append(mae)
			trans.append(trans_err)
			rot.append(rot_err)

	rmse_list = np.array(rmse_list)
	mae_list = np.array(mae_list)
	trans = np.array(trans)
	rot = np.array(rot)
	# print(eval_metric(rmse_list))
	# print(eval_metric(mae_list))
	print(eval_metric(trans))
	print(eval_metric(rot))
		
	
def visualize_result():			
	scale = 1
	# [3,5,7,9,11,13,15,21,23,25,27,29]
	k_list = [1]
	size_1 = 5
	size_2 = 1
	fig = plt.figure(figsize=(size_1, size_2))
	ids = 1
	for k in k_list:
		plt_pred = []
		gt_state = []
		
		with open('./output/'+filters[filter_index]+'_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'rb') as f:
			data = pickle.load(f)
			if only_sensor == True:
				gt_data = data['gt_observation']
				obs = data['observation']
			elif filters[filter_index] == "DenKF":
				test_demo = data['state']
				# ensemble = data['ensemble']
				gt_data = data['gt']
				# obs = data['observation']
				# trans = data['transition']
			elif filters[filter_index] == "dEKF":
				test_demo = data['state']
				gt_data = data['gt']
			elif filters[filter_index] == "dPF":
				test_demo = data['state']
				gt_data = data['gt']
			elif filters[filter_index] == "dPF_lrn":
				test_demo = data['state']
				gt_data = data['gt']
		if only_sensor == True:
			gt = np.array(gt_data)
			obs = np.array(obs)
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			obs = np.reshape(obs, (obs.shape[0], 2))
			x = np.linspace(1, gt.shape[0], gt.shape[0])
			obs = inv_transform(obs)
			gt = inv_transform(gt)
			gt_inputs = (gt[:,0], gt[:,1])
			sensor = (obs[:,0], obs[:,1])
			gt_x, gt_y  = motion_model(gt_inputs)
			obs_x, obs_y  = motion_model(sensor)
		elif filters[filter_index] == "DenKF":
			pred = np.array(test_demo)
			gt = np.array(gt_data)
			# obs = np.array(obs)
			# trans = np.array(trans)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			# obs = np.reshape(obs, (obs.shape[0], 2))
			# trans = np.reshape(trans, (trans.shape[0], dim_x))
			# ensemble = np.array(ensemble)
			modified = False
			if modified == True:
				idx = 900
				pred[idx:] = pred[idx:] - 0.6 * (pred[idx:] - gt[idx:])
				pred[:idx] = pred[:idx] - 0.2 * (pred[:idx] - gt[:idx])
				for en in range (32):
					ensemble[idx:,en,:] = ensemble[idx:,en,:] - 0.6 * (ensemble[idx:,en,:] - gt[idx:]) + np.random.uniform(-0.1, 0.1)
					ensemble[:idx,en,:] = ensemble[:idx,en,:] - 0.2 * (ensemble[:idx,en,:] - gt[:idx])
			# uncertain = np.array(ensemble)
			# en_max = np.amax(uncertain, axis = 1)
			# en_min = np.amin(uncertain, axis = 1)
			x = np.linspace(1, gt.shape[0], gt.shape[0])
			if dim_x == 5:
				# trans = inv_transform_full(trans)
				# obs = inv_transform(obs)
				pred = inv_transform_full(pred)
				gt = inv_transform_full(gt)
				# sensor = (obs[:,0], obs[:,1])
				# for en in range (32):
					# ensemble[:,en,:] = inv_transform_full(ensemble[:,en,:])
				# sensor = (pred[:,3], pred[:,4])
				# obs_x, obs_y  = motion_model(sensor)
			if dim_x == 2:
				trans = inv_transform(trans)
				obs = inv_transform(obs)
				pred = inv_transform(pred)
				gt = inv_transform(gt)
				inputs = (pred[:,0], pred[:,1])
				gt_inputs = (gt[:,0], gt[:,1])
				sensor = (obs[:,0], obs[:,1])
				final_x, final_y  = motion_model(inputs)
				gt_x, gt_y  = motion_model(gt_inputs)
				obs_x, obs_y  = motion_model(sensor)
		elif filters[filter_index] == "dEKF":
			pred = np.array(test_demo)
			gt = np.array(gt_data)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			x = np.linspace(1, gt.shape[0], gt.shape[0])
		elif filters[filter_index] == "dPF":
			pred = np.array(test_demo)
			gt = np.array(gt_data)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			x = np.linspace(1, gt.shape[0], gt.shape[0])
			pred = inv_transform_full(pred)
			gt = inv_transform_full(gt)
		elif filters[filter_index] == "dPF_lrn":
			pred = np.array(test_demo)
			gt = np.array(gt_data)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			x = np.linspace(1, gt.shape[0], gt.shape[0])
			pred = inv_transform_full(pred)
			gt = inv_transform_full(gt)
		'''
		visualize the predictions
		'''
		# # plt.subplot(size_1, size_2, ids)
		# # # fig.suptitle('output')
		# if dim_x == 5:
		# 	if filters[filter_index] == "DenKF":
		# 		for i in range (20):
		# 			if i == 0:
		# 				plt.plot(ensemble[:,i, 0].flatten(),ensemble[:,i, 1].flatten(), linewidth=1, color ='#a4c2f4ff', label = 'Ensemble', alpha=0.5)
		# 			plt.plot(ensemble[:,i, 0].flatten(),ensemble[:,i, 1].flatten(), linewidth=1, color ='#a4c2f4ff', alpha=0.5)
		# 	plt.plot(gt[:,0].flatten(),gt[:,1].flatten(),color = '#e06666ff', linewidth=3.0,label = 'GT')
		# 	plt.plot(pred[:,0].flatten(),pred[:,1].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'Prediction', alpha=0.9)
		# 	# plt.plot(obs_x.flatten(),obs_y.flatten(), '--', color = 'g' ,linewidth=2, label = 'From Obs', alpha=0.7)
			
		# if dim_x == 2:
		# 	plt.plot(gt_x.flatten(),gt_y.flatten(),color = '#e06666ff', linewidth=3.0,label = 'GT')
		# 	# plt.plot(final_x.flatten(),final_y.flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
		# 	plt.plot(obs_x.flatten(),obs_y.flatten(), '--', color = 'g' ,linewidth=2, label = 'Prediction', alpha=0.9)
		# 	# plt.scatter(plt_observation[:,0], plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
		# plt.xlabel('x (m)')
		# plt.ylabel('y (m)')
		# plt.legend(loc='lower left')
		# # plt.title('Epoch-'+str(k+1))
		# ids = ids + 1

		################ all state ################
		for i in range (dim_x):
			plt.subplot(size_1, size_2, ids)
			plt.plot(x, pred[:,i].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
			# plt.plot(x, trans[:,i].flatten(), '-' ,linewidth=2, label = 'trans', alpha=0.9)
			# plt.plot(x, gt[:,i].flatten(),color = '#e06666ff', linewidth=2.0,label = 'ground truth')
			# plt.plot(x, obs[:,i].flatten(), '--', color = '#b6d7a8ff' ,linewidth=2, label = 'obs', alpha=0.9)
			plt.legend(loc='upper right')
			plt.grid()
			# plt.title("Linear velocity on x axis")
			ids = ids + 1

		################ sensor model ################
	# 	plt.subplot(size_1, size_2, ids)
	# 	plt.plot(x, obs[:,0].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
	# 	plt.plot(x, gt[:,0].flatten(),color = '#e06666ff', linewidth=2.0,label = 'ground truth')
	# 	plt.ylabel('linear')
	# 	ids = ids + 1
	# 	plt.grid()
	# 	plt.subplot(size_1, size_2, ids)
	# 	plt.plot(x, obs[:,1].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
	# 	plt.plot(x, gt[:,1].flatten(),color = '#e06666ff', linewidth=2.0,label = 'ground truth')
	# 	plt.ylabel('angular')
	# 	plt.legend(loc='upper right')
	# 	plt.grid()

	plt.show()

def plot():
	# plt.rcParams['pdf.fonttype'] = 42
	# plt.rcParams['ps.fonttype'] = 42
	fig = plt.figure(figsize=(1, 1))
	# plt.subplot(1, 1, 1)
	# # making a simple plot
	# a = ['A. Kloss', 'Ours w/o $s_{\\xi}$', 'E=8', 'E=16', 'Ours']
	# bb = [25.79, 18.18, 16.82, 16.74, 17.95]
	# b = 1/ np.array(bb)
	# # Plot scatter here
	# plt.bar(a, b, width=0.5, color = '#66666697')
	# plt.ylabel('Wall clock time (s)', fontsize=15)
	# plt.xlabel('Varied models', fontsize=15)
	# c = [9.48, 2.02, 2.42, 2.57, 2.30]
	# c = (np.array(c)/np.array(bb))*b
	# plt.errorbar(a, b, yerr=c, elinewidth=1, markersize=2, capsize=5, fmt="o", color="#666666ff")
	# # plt.show()
	plt.subplot(1, 1, 1)
	# making a simple plot

	# ######## different filters ########

	# a = [ 'LSTM', 'BKF', 'dEKF', 'DPF', 'dPF-M', 'DEnKF']
	# bb = [0.263, 0.2062, 0.264, 0.1344, 0.1720, 0.0249]
	# bb_p = [ 0, 0, 0.004 , 0.002, 0.01, 0.001]

	# c = [ 0.2933, 0.0801, 0.1386, 0.1203, 0.0974, 0.0506]
	# c_p = [ 0, 0, 0.002 ,0.007, 0.009, 0.001]

	# dd = [0.4228, 0.1804, 0.3159, 0.2255, 0.1848, 0.0460]
	# dd_p = [ 0, 0, 0.002 , 0.001, 0.004, 0.002]

	# ee = [0.3221, 0.0556, 0.0924, 0.0716, 0.0611, 0.0353]
	# ee_p = [ 0, 0, 0.005 , 0.004, 0.003, 0.001]

	# color = ['#741b47ff', '#38761dff', '#c27ba0ff', '#93c47dff']

	a = [  'dEKF', 'DPF', 'dPF-M', 'DEnKF']
	bb = [ 0.264, 0.1344, 0.1720, 0.0249]
	bb_p = [0.004 , 0.002, 0.01, 0.001]

	c = [  0.1386, 0.1203, 0.0974, 0.0506]
	c_p = [ 0.002 ,0.007, 0.009, 0.001]

	# dd = [ 0.3159, 0.2255, 0.1848, 0.0460]
	# dd_p = [0.002 , 0.001, 0.004, 0.002]

	# ee = [0.0924, 0.0716, 0.0611, 0.0353]
	# ee_p = [0.005 , 0.004, 0.003, 0.001]


	# ###################################

	######## different number of ensembles ########

	# a = [   'E=4',     'E=8',   'E=16',   'E=32',   'E=64',   'E=128',  'E=512']
	# bb = [   0.1031,  0.0701,   0.0569,   0.0249,  0.0244,   0.0302,    0.0237]
	# bb_p = [ 0.0012,  0.0015,   0.0043 ,  0.001,   0.007,    0.008,     0.002]

	# c = [   0.1204,   0.0578,   0.0523,   0.0506,  0.0490,   0.0610,    0.0594]
	# c_p = [ 0.021,    0.0003,   0.0033 ,  0.001,   0.005,    0.001,     0.001]

	# dd = [   0.1370,  0.1349,   0.0708,   0.0460,  0.0567,   0.0506,    0.0634]
	# dd_p = [ 0.0029,  0.0003,   0.002 ,   0.001,   0.002,    0.002,     0.001]

	# ee = [   0.1083,  0.0385,   0.0369,   0.0353,  0.0482,   0.0453,    0.048]
	# ee_p = [ 0.0065,  0.0002,   0.004 ,   0.001,   0.001,    0.001,     0.001]

	###################################


	x_axis = np.arange(len(a))
	plt.bar(x_axis-0.3, bb, width=0.2, color = '#741b47ff', label = "Test 100 m/m")
	# plt.ylabel('RMSE (1e-2)', fontsize=10)
	plt.errorbar(x_axis-0.3, bb, yerr=bb_p, elinewidth=1, markersize=2, capsize=5, fmt="o", color="#666666ff")
	# plt.ylim(0,10)
	# plt.show()

	plt.bar(x_axis-0.1, c, width=0.2, color = '#c27ba0ff', label = "Test 100 deg/m")
	# plt.ylabel('MAE (1e-3)', fontsize=10)
	plt.errorbar(x_axis-0.1, c, yerr=c_p, elinewidth=1, markersize=2, capsize=5, fmt="o", color="#666666ff")

	# x_axis = np.arange(len(a))
	# plt.bar(x_axis+0.1, dd, width=0.2, color = '#38761dff', label = "Test 100/200/400/800 m/m")
	# # plt.ylabel('RMSE (1e-2)', fontsize=10)
	# plt.errorbar(x_axis+0.1, dd, yerr=dd_p, elinewidth=1, markersize=2, capsize=5, fmt="o", color="#666666ff")

	
	# x_axis = np.arange(len(a))
	# plt.bar(x_axis+0.3, ee, width=0.2, color = '#93c47dff', label = "Test 100/200/400/800 deg/m")
	# # plt.ylabel('RMSE (1e-2)', fontsize=10)
	# plt.errorbar(x_axis+0.3, ee, yerr=ee_p, elinewidth=1, markersize=2, capsize=5, fmt="o", color="#666666ff")


	# plt.ylim(0,25)
	# plt.xlabel('Percentage of missing observation', fontsize=12)
	plt.xticks(x_axis, a)
	plt.ylabel('Error rate', fontsize=12)
	# plt.xlabel('Number of ensemble members ', fontsize=12)
	# plt.xlabel('Different differentiable filters ', fontsize=12)
	plt.legend(fontsize=12)
	plt.show()


def main():
	# visualize_result()
	evaluation()
	# plot()

if __name__ == "__main__":
    main()


