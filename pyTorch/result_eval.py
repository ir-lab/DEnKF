import pickle
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['savefig.dpi'] = 500
import numpy as np
from einops import rearrange, repeat
import pdb
import os
import pickle
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math


dim_x = 7

def quaternion_to_euler(w, x, y, z):

	t0 = 2 * (w * x + y * z)
	t1 = 1 - 2 * (x * x + y * y)
	X = math.atan2(t0, t1)

	t2 = 2 * (w * y - z * x)
	t2 = 1 if t2 > 1 else t2
	t2 = -1 if t2 < -1 else t2
	Y = math.asin(t2)
	
	t3 = 2 * (w * z + x * y)
	t4 = 1 - 2 * (y * y + z * z)
	Z = math.atan2(t3, t4)

	return X, Y, Z


def visualize_result():
	k_list = ['178800']# 178800 992100
	size_1 = 3
	size_2 = 1
	fig = plt.figure(figsize=(size_1, size_2))
	ids = 1
	for k in k_list:
		with open('result_test.pkl', 'rb') as f:
			data = pickle.load(f)
			test_demo = data['state']
			gt_data = data['gt']

			pred = np.squeeze(np.array(test_demo))
			gt = np.squeeze(gt_data)

			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			x = np.linspace(1, gt.shape[0], gt.shape[0])

		'''
		visualize the predictions
		'''
		############## all state ################
		for i in range (3):
			plt.subplot(size_1, size_2, ids)
			plt.plot(x, pred[:,i].flatten(), '--',color = 'b',linewidth=1, alpha=1, label = 'Pred')
			plt.plot(x, gt[:,i].flatten(), color = 'r',linewidth=1,label = 'Mocap-'+str(i+1))
			ids = ids + 1
	plt.show()


def main():
	visualize_result()

if __name__ == "__main__":
    main()


