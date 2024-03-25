#!/usr/bin/env python

import os
import csv
import rospkg
import rospy
import geometry_msgs.msg
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from std_msgs.msg import Float64MultiArray

import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

# NUM_BINS = 20
initial_fthresh = 2
param_fthresh = 2
path = ''

def fit_gauss(df, key, metric, num_bins):
	
	samples = df[key][(np.abs(stats.zscore(df[key])) < param_fthresh)]

	print("{:^5s} | Num samples after filtering: {}".format(key, len(samples)))

	mu = np.mean(samples)
	sigma = np.std(samples, ddof=1)
	pad = np.tanh(3*sigma)*0.2 # Use tanh to limit the padding at 0.2
	x = np.array(np.linspace(min(samples)-pad, max(samples)+pad, 100)).reshape(-1,1)

	# Normalise and scale PDF to the same height as histogram 
	h, _ = np.histogram(samples, bins=num_bins)
	y = stats.norm.pdf(x, mu, sigma)/max(stats.norm.pdf(x, mu, sigma))*max(h)

	return {'key':key, 'metric':metric, 'samples':samples, 'x':x, 'y':y, 'mu':mu, 'sigma':sigma}
			

def publish_params(gauss):

	pub = rospy.Publisher('/extrinsic_calib_param', Float64MultiArray, queue_size=10)
	rate = rospy.Rate(10)
	msg = Float64MultiArray()    

	means, stdevs = [], []
	for param in gauss:
		mean = param['mu']
		stdev = param['sigma']
		means.append(mean)
		stdevs.append(stdev)

	# Publish data in the order: roll pitch yaw x y z 
	msg.data = means 

	while not rospy.is_shutdown():
		pub.publish(msg)
		rate.sleep()

def visualise_results(gauss, nbins_list, degree):

	colors = ["violet","thistle","royalblue","indianred","turquoise","lightslategray"]

	# Setup Subplot
	fig, ax = plt.subplots(2,3,figsize=(12,7))
	fig.tight_layout(pad=3.0) # space out the plots a bit
	fig.subplots_adjust(top=0.93) # Adjust spacing at the top for the suptitle
	fig.suptitle('Extrinsic Parameter Results')
	row = [0,0,0,1,1,1]
	col = [0,1,2,0,1,2]
	
	for idx, param in enumerate(gauss):
		x = param['x']
		y = param['y']
		mu = param['mu']
		stdev = param['sigma']

		# Plot PDF
		r = row[idx]
		c = col[idx]
		ax[r,c].hist(param['samples'], bins=nbins_list[idx], alpha=0.6, color=colors[idx])
		ax[r,c].plot(x, y, 'b-', color="black")
		ax[r,c].set_xlabel(param['key'] + ' (' + param['metric'] + ')')
		ax[r,c].set_ylabel('Frequency')
		ax[r,c].set_ylim(0, max(y)+float(max(y))/5)
		ax[r,c].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		if degree and idx > 2:
			ax[r,c].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		else:
			ax[r,c].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

		# Annotate max value of highest weighted gaussian
		bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
		arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=60")
		kw = dict(xycoords='data',textcoords="axes fraction",
				  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")

		if degree and idx > 2:
			ax[r,c].annotate(param['key'] + "=% .3f\nstd=% .3f" % (mu, stdev), xy=(mu, max(y)), xytext=(0.94,0.96), **kw)
		else:
			ax[r,c].annotate(param['key'] + "=% .5f\nstd=% .5f" % (mu, stdev), xy=(mu, max(y)), xytext=(0.94,0.96), **kw)
		
	plt.show(block=False)

if __name__ == '__main__':
	rospy.init_node('visualise_results', anonymous=True)
	rospy.loginfo("Starting visualise_results")
	path = rospy.get_param("~csv")
	degree = rospy.get_param("~degree")
	bin_width_trans = rospy.get_param("~trans_binwidth") # in metres
	bin_width_rot = rospy.get_param("~rot_binwidth_deg")*np.pi/180 # deg to rads

	if not os.path.exists(path):
		raise Exception('GAUSS FITTING - No file found at: {}'.format(path))
		exit()

	rospy.loginfo("Opening file at: " + path) 
	rospy.loginfo("Using degrees for rotation")

	# Read data and fit GMM
	df_orig = pandas.read_csv(path)
	df = df_orig.copy()

	# Initial filtering of general outliers
	params = ['roll','pitch','yaw','x','y','z']
	for p in params:
		df = df[(np.abs(stats.zscore(df[p])) < initial_fthresh)]

	num_bins_x = int(np.ceil((df['x'].max()-df['x'].min())/bin_width_trans))
	num_bins_y = int(np.ceil((df['y'].max()-df['y'].min())/bin_width_trans))
	num_bins_z = int(np.ceil((df['z'].max()-df['z'].min())/bin_width_trans))
	num_bins_roll = int(np.ceil((df['roll'].max()-df['roll'].min())/bin_width_rot))
	num_bins_pitch = int(np.ceil((df['pitch'].max()-df['pitch'].min())/bin_width_rot))
	num_bins_yaw = int(np.ceil((df['yaw'].max()-df['yaw'].min())/bin_width_rot))
	nbins_list = [num_bins_roll, num_bins_pitch, num_bins_yaw, num_bins_x, num_bins_y, num_bins_z]
	# print(nbins_list)

	print("\nTotal number of samples: {}".format(len(df)))

	gauss = []
	if not degree:
		gauss.append(fit_gauss(df, 'roll', 'rad', num_bins=num_bins_roll))
		gauss.append(fit_gauss(df, 'pitch', 'rad', num_bins=num_bins_pitch))
		gauss.append(fit_gauss(df, 'yaw', 'rad', num_bins=num_bins_yaw))  
	else:
		df['roll'] = df_orig['roll']*180/np.pi
		df['pitch'] = df_orig['pitch']*180/np.pi
		df['yaw'] = df_orig['yaw']*180/np.pi
		gauss.append(fit_gauss(df, 'roll', 'deg', num_bins=num_bins_roll))
		gauss.append(fit_gauss(df, 'pitch', 'deg', num_bins=num_bins_pitch))
		gauss.append(fit_gauss(df, 'yaw', 'deg', num_bins=num_bins_yaw))

	gauss.append(fit_gauss(df, 'x', 'm', num_bins=num_bins_x))
	gauss.append(fit_gauss(df, 'y', 'm', num_bins=num_bins_y))
	gauss.append(fit_gauss(df, 'z', 'm', num_bins=num_bins_z))

	print("\n")     
	visualise_results(gauss, nbins_list, degree)    

	try:
		publish_params(gauss)
	except rospy.ROSInterruptException:
		pass