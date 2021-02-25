#!/usr/bin/env python3

import os 
import cv2
import math
import time
import struct
import pickle
import numpy as np
from pycpd import DeformableRegistration
import operator
from PIL import Image
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# ROS
import rospy
import roslib
import rospkg
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from medical_msgs.msg import *

# Torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

# Obman
from handobjectdatasets.queries import TransQueries, BaseQueries
from handobjectdatasets.viz2d import visualize_joints_2d_cv2
from mano_train.exputils import argutils
from mano_train.netscripts.reload import reload_model
from mano_train.visualize import displaymano
from mano_train.demo.preprocess import prepare_input, preprocess_frame
from mano_train.demo.attention import AttentionHook

class mano_prediction(object):
	def __init__(self):
		self.bridge = CvBridge()

		# right, left
		self.mode = "right"

		# Model path
		r = rospkg.RosPack()
		self.path = r.get_path("obman_prediction")
		mano_model_name = "mano/MANO_RIGHT.pkl"

		hands_only_model_name = "hands_only/opt.pkl"
		hands_only_checkpoint = os.path.join(self.path, "weight/hands_only/checkpoint.pth.tar")
		with open(os.path.join(self.path, "weight", hands_only_model_name), "rb") as opt_fh:
			opts_hands = pickle.load(opt_fh)

		# Initialize network
		self.model_hands = reload_model(hands_only_checkpoint, opts_hands)
		self.model_hands.eval()

		# Load faces of hand
		with open(os.path.join(self.path, "weight", mano_model_name), "rb") as p_f:
			mano_right_data = pickle.load(p_f, encoding="latin1")
			self.faces = mano_right_data["f"]
		
		# Publisher
		self.predict_left_2D = rospy.Publisher("mano/prediction_left_2D", Image, queue_size = 1)
		self.predict_right_2D = rospy.Publisher("mano/prediction_right_2D", Image, queue_size = 1)
		self.pose_pub = rospy.Publisher("mano/hand_object_pose", HandObjectPoseArray, queue_size=1)

		# Mssage filter 
		image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
		depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
		ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 1)
		ts.registerCallback(self.callback)

		# Camera info
		info = rospy.wait_for_message('camera/color/camera_info', CameraInfo)
		self.fx = info.P[0]
		self.fy = info.P[5]
		self.cx = info.P[2]
		self.cy = info.P[6]

	def callback(self, rgb, depth):
		# Ros image to cv2
		try:
			cv_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
			cv_depth = self.bridge.imgmsg_to_cv2(depth, "16UC1")
		except CvBridgeError as e:
			print(e)
		
		# Image prepare
		frame = preprocess_frame(cv_image)
		input_image = prepare_input(frame)
		hand_crop = cv2.resize(np.array(frame), (256, 256))

		# Define msgs
		position_msgs = HandObjectPoseArray()
		position_msgs.header = rgb.header
		position_msgs.header.frame_id = "camera_link"

	# ================= noflip (left-hand) ====================
		if self.mode == "left":

			noflip_hand_image = prepare_input(hand_crop, flip_left_right=False)
			noflip_2D_output = self.forward_pass_2d(noflip_hand_image)
			
			noflip_image = deepcopy(hand_crop)
			if "joints2d" in noflip_2D_output:
				joints2d = noflip_2D_output["joints2d"].cpu().detach().numpy()[0]
				noflip_image = visualize_joints_2d_cv2(noflip_image, joints2d)

		# ================= Hand Point =================
			for i in range(21):
				joints2d[i][0] = int(joints2d[i][0] * 2.5)
				joints2d[i][1] = int(joints2d[i][1] * 1.875)

				if (int(joints2d[i][0]) < 640 and int(joints2d[i][1]) < 480):
					zc = cv_depth[int(joints2d[i][1]), int(joints2d[i][0])]/1000
					hand_rx, hand_ry, hand_rz = self.getXYZ(joints2d[i][0] /1.0 , joints2d[i][1], zc/1.0 )
					location = "finger_" + str(i) 
					position_msgs = self.append_to_array(position_msgs, hand_rx, hand_ry, hand_rz, location)
	
		# ================= Publisher ==================
			self.predict_left_2D.publish(self.bridge.cv2_to_imgmsg(noflip_image, "8UC3"))
			self.pose_pub.publish(position_msgs)

	# ================= flip (right-hand) ====================
		if self.mode == "right":

			flip_hand_image = prepare_input(hand_crop, flip_left_right=True)
			flip_2D_output = self.forward_pass_2d(flip_hand_image)

			flip_image = deepcopy(np.flip(hand_crop, axis=1))
			if "joints2d" in flip_2D_output:
				joints2d = flip_2D_output["joints2d"].cpu().detach().numpy()[0]
				flip_image = visualize_joints_2d_cv2(flip_image, joints2d)

	# ================= Hand Point =================
			for i in range(21):
				joints2d[i][0] = int(640 - joints2d[i][0]*2.5)
				joints2d[i][1] = int(joints2d[i][1]*1.875)

				if (int(joints2d[i][0]) < 640 and int(joints2d[i][1]) < 480):
					zc = cv_depth[int(joints2d[i][1]), int(joints2d[i][0])]/1000
					hand_rx, hand_ry, hand_rz = self.getXYZ(joints2d[i][0] /1.0 , joints2d[i][1], zc/1.0 )
					location = "finger_" + str(i) 
					position_msgs = self.append_to_array(position_msgs, hand_rx, hand_ry, hand_rz, location)

	# ================= Publisher ==================
			predict_image = cv2.flip(flip_image, 1)
			self.predict_right_2D.publish(self.bridge.cv2_to_imgmsg(predict_image, "8UC3"))
			self.pose_pub.publish(position_msgs)
		
		rgb = None
		depth = None

	def getXYZ(self, x, y, zc):
		
		x = float(x)
		y = float(y)
		zc = float(zc)
		inv_fx = 1.0/self.fx
		inv_fy = 1.0/self.fy
		x = (x - self.cx) * zc * inv_fx
		y = (y - self.cy) * zc * inv_fy 
		z = zc 

		return z, -1*x, -1*y

	def append_to_array(self, pose_array, rx, ry, rz, location):
		
		pose = HandObjectPose()
		pose.location = location
		pose.pose.position.x = rx
		pose.pose.position.y = ry
		pose.pose.position.z = rz
		pose_array.pose_array.append(pose)

		return pose_array


	def visualize(self, iteration, error, X, Y, ax):
		
		plt.cla()
		ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
		ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
		iteration = 9
		plt.text(0.87, 0.92, 'Iteration: {:d}'.format(iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
		ax.legend(loc='upper left', fontsize='x-large')
		plt.draw()
		plt.pause(0.001)

	def forward_pass_2d(self, input_image, pred_obj=True):
	
		sample = {}
		sample[TransQueries.images] = input_image
		sample[BaseQueries.sides] = ["left"]
		sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
		sample["root"] = "wrist"
		if pred_obj:
			sample[TransQueries.objpoints3d] = input_image.new_ones((1, 600, 3)).float()
		_, results, _ = self.model_hands.forward(sample, no_loss=True)

		return results

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	
	

if __name__ == '__main__': 
	rospy.init_node('mano_prediction',anonymous=False)
	mano_prediction = mano_prediction()
	rospy.on_shutdown(mano_prediction.onShutdown)
	rospy.spin()
