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
from medical_msgs.srv import *

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

# CPD
from functools import partial
import matplotlib.pyplot as plt
from pycpd import DeformableRegistration
from pycpd import RigidRegistration
from pycpd import AffineRegistration

class obman_prediction(object):
	def __init__(self):
		self.bridge = CvBridge()

		# right, left
		self.mode = "right"

		# Switch
		self.switch = False

		# Model path
		r = rospkg.RosPack()
		self.path = r.get_path("obman_prediction")
		mano_model_name = "mano/MANO_RIGHT.pkl"

		obman_model_name = "obman/opt.pkl"
		obman_checkpoint = os.path.join(self.path, "weight/obman/checkpoint.pth.tar")
		with open(os.path.join(self.path, "weight", obman_model_name), "rb") as opt_fo:
			opts_obman = pickle.load(opt_fo)

		hands_only_model_name = "hands_only/opt.pkl"
		hands_only_checkpoint = os.path.join(self.path, "weight/hands_only/checkpoint.pth.tar")
		with open(os.path.join(self.path, "weight", hands_only_model_name), "rb") as opt_fh:
			opts_hands = pickle.load(opt_fh)

		# Initialize network
		self.model_obman = reload_model(obman_checkpoint, opts_obman)
		self.model_obman.eval()

		self.model_hands = reload_model(hands_only_checkpoint, opts_hands)
		self.model_hands.eval()

		# Load faces of hand
		with open(os.path.join(self.path, "weight", mano_model_name), "rb") as p_f:
			mano_right_data = pickle.load(p_f, encoding="latin1")
			self.faces = mano_right_data["f"]
		
		# Publisher
		# self.predict_blend_img_hand = rospy.Publisher("obman/prediction_blend_img_hand", Image, queue_size = 1)
		# self.predict_blend_img_atlas = rospy.Publisher("obman/prediction_blend_img_atlas", Image, queue_size = 1)
		self.predict_left_3D = rospy.Publisher("obman/prediction_left_3D", Image, queue_size = 1)
		self.predict_right_3D = rospy.Publisher("obman/prediction_right_3D", Image, queue_size = 1)
		self.predict_left_2D = rospy.Publisher("obman/prediction_left_2D", Image, queue_size = 1)
		self.predict_right_2D = rospy.Publisher("obman/prediction_right_2D", Image, queue_size = 1)
		self.pose_pub = rospy.Publisher("obman/hand_object_pose", HandObjectPoseArray, queue_size=1)


		# Add attention map
		# self.attention_hand = AttentionHook(self.model_obman.module.base_net)
		# if hasattr(self.model_obman.module, "atlas_base_net"):
		# 	self.attention_atlas = AttentionHook(self.model_obman.module.atlas_base_net)
		# 	self.has_atlas_encoder = True
		# else:
		# 	self.has_atlas_encoder = False


		# service
		self.predict_switch = rospy.Service("~predict_switch_server", obman_switch, self.switch_callback)

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
		if not self.switch:
			return

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

		# Attention map 
		# blend_img_hand = self.attention_hand.blend_map(frame)
		# self.predict_blend_img_hand.publish(self.bridge.cv2_to_imgmsg(blend_img_hand, "bgr8"))

		# if self.has_atlas_encoder:
		# 	blend_img_atlas = self.attention_atlas.blend_map(frame)
		# 	self.predict_blend_img_atlas.publish(self.bridge.cv2_to_imgmsg(blend_img_atlas, "bgr8"))

	# ================= noflip (left-hand) ====================
		if self.mode == "left":

			noflip_fig = plt.figure(figsize=(4, 4))
			noflip_hand_image = prepare_input(hand_crop, flip_left_right=False)
			noflip_3D_output = self.forward_pass_3d(noflip_hand_image)
			noflip_2D_output = self.forward_pass_2d(noflip_hand_image)
			
			noflip_image = deepcopy(hand_crop)
			if "joints2d" in noflip_2D_output:
				joints2d = noflip_2D_output["joints2d"].cpu().detach().numpy()[0]
				noflip_image = visualize_joints_2d_cv2(noflip_image, joints2d)

			noflip_verts = noflip_3D_output["verts"].cpu().detach().numpy()[0]
			noflip_ax = noflip_fig.add_subplot(1, 1, 1, projection="3d")
			displaymano.add_mesh(noflip_ax, noflip_verts, self.faces, flip_x=False)

			if "objpoints3d" in noflip_3D_output:
				objverts = noflip_3D_output["objpoints3d"].cpu().detach().numpy()[0]
				displaymano.add_mesh(noflip_ax, objverts, noflip_3D_output["objfaces"], flip_x=False, c="r")

		# ================= Coherent Point Drift for matching =================	
			target = np.array([joints2d[0], joints2d[4], joints2d[8], joints2d[12], joints2d[16], joints2d[20]])
			source = np.array([noflip_verts[279][:2], noflip_verts[730][:2], noflip_verts[317][:2], noflip_verts[443][:2], noflip_verts[556][:2], noflip_verts[673][:2]])

			reg = AffineRegistration(**{'X': target, 'Y': source})
			TY, _ = reg.register()
			self.B, self.t = AffineRegistration.get_registration_parameters(reg)

		# ================= Hand Point =================
			for i in range(21):
				joints2d[i][0] = int(joints2d[i][0] * 2.5)
				joints2d[i][1] = int(joints2d[i][1] * 1.875)

				if (0 < int(joints2d[i][0]) < 640 and 0 < int(joints2d[i][1]) < 480):
					zc = cv_depth[int(joints2d[i][1]), int(joints2d[i][0])]/1000
					hand_rx, hand_ry, hand_rz = self.getXYZ(joints2d[i][0] /1.0 , joints2d[i][1], zc/1.0 )
					location = "finger"
					position_msgs = self.append_to_array(position_msgs, hand_rx, hand_ry, hand_rz, location)

		# ================= Object Point =================
			for index, obj_points in enumerate(objverts):
				obj_point = AffineRegistration.transform_point_cloud(self, obj_points[:2])
				obj_point = obj_point.astype(np.float32)
				obj_visualize = obj_point.astype(np.float32)
				vis_obj = tuple(obj_visualize[0])

				obj_point[0][0] = int(obj_point[0][0] * 2.5)
				obj_point[0][1] = int(obj_point[0][1] * 1.875)
				display_obj = tuple(obj_point[0])
				cv2.circle(cv_image, display_obj, 1, (0, 255, 255), 4)
			
				if index == 592:
					location = "left_bottom"					
					cv2.circle(noflip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 43:		
					location = "right_bottom"
					cv2.circle(noflip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 413:
					location = "center_bottom"
					cv2.circle(noflip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 438:
					location = "center"
					cv2.circle(noflip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 195:
					location = "left_top"
					cv2.circle(noflip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 379:
					location = "center_top"
					cv2.circle(noflip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 316:
					location = "right_bottom"
					cv2.circle(noflip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)		

				elif index == 455:
					location = "center_right"
					cv2.circle(noflip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 431:		
					location = "center_left"
					cv2.circle(noflip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)
	
		# ================= Publisher ==================
			noflip_fig.canvas.draw()
			w, h = noflip_fig.canvas.get_width_height()
			predict_noflip = np.fromstring(noflip_fig.canvas.tostring_argb(), dtype=np.uint8)
			predict_noflip.shape = (w, h, 4)

			converted_img = cv2.cvtColor(predict_noflip, cv2.COLOR_BGRA2BGR)
			position_msgs.image = self.bridge.cv2_to_imgmsg(converted_img, "8UC3")

			self.predict_left_2D.publish(self.bridge.cv2_to_imgmsg(noflip_image, "bgr8"))
			self.predict_left_3D.publish(self.bridge.cv2_to_imgmsg(predict_noflip, "8UC4"))
			self.pose_pub.publish(position_msgs)


	# ================= flip (right-hand) ====================
		if self.mode == "right":

			flip_fig = plt.figure(figsize=(4, 4))																
			flip_hand_image = prepare_input(hand_crop, flip_left_right=True)
			flip_3D_output = self.forward_pass_3d(flip_hand_image)
			flip_2D_output = self.forward_pass_2d(flip_hand_image)

			flip_image = deepcopy(np.flip(hand_crop, axis=1))
			if "joints2d" in flip_2D_output:
				joints2d = flip_2D_output["joints2d"].cpu().detach().numpy()[0]
				flip_image = visualize_joints_2d_cv2(flip_image, joints2d)

			flip_verts = flip_3D_output["verts"].cpu().detach().numpy()[0]
			flip_ax = flip_fig.add_subplot(1, 1, 1, projection="3d")
			displaymano.add_mesh(flip_ax, flip_verts, self.faces, flip_x=True)

			if "objpoints3d" in flip_3D_output:
				objverts = flip_3D_output["objpoints3d"].cpu().detach().numpy()[0]
				displaymano.add_mesh(flip_ax, objverts, flip_3D_output["objfaces"], flip_x=True, c="r")

	# ================= Coherent Point Drift for matching =================	
			target = np.array([joints2d[0], joints2d[4], joints2d[8], joints2d[12], joints2d[16], joints2d[20]])
			source = np.array([flip_verts[279][:2], flip_verts[730][:2], flip_verts[317][:2], flip_verts[443][:2], flip_verts[556][:2], flip_verts[673][:2]])

			reg = AffineRegistration(**{'X': target, 'Y': source})
			TY, _ = reg.register()
			self.B, self.t = AffineRegistration.get_registration_parameters(reg)

	# ================= Hand Point =================
			for i in range(21):
				joints2d[i][0] = int(640 - joints2d[i][0]*2.5)
				joints2d[i][1] = int(joints2d[i][1]*1.875)

				if (0 < int(joints2d[i][0]) < 640 and 0 < int(joints2d[i][1]) < 480):
					zc = cv_depth[int(joints2d[i][1]), int(joints2d[i][0])]/1000
					hand_rx, hand_ry, hand_rz = self.getXYZ(joints2d[i][0] /1.0 , joints2d[i][1], zc/1.0 )
					location = "finger" 
					position_msgs = self.append_to_array(position_msgs, hand_rx, hand_ry, hand_rz, location)

	# ================= Object Point =================
			for index, obj_points in enumerate(objverts):
				obj_point = AffineRegistration.transform_point_cloud(self, obj_points[:2])
				obj_point = obj_point.astype(np.float32)
				obj_visualize = obj_point.astype(np.float32)
				vis_obj = tuple(obj_visualize[0])

				obj_point[0][0] = int(640 - obj_point[0][0] * 2.5)
				obj_point[0][1] = int(obj_point[0][1] * 1.875)
				display_obj = tuple(obj_point[0])
				cv2.circle(cv_image, display_obj, 1, (0, 255, 255), 4)
			
				if index == 592:
					location = "left_bottom"					
					cv2.circle(flip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)
					
				elif index == 43:		
					location = "right_bottom"
					cv2.circle(flip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 413:
					location = "center_bottom"
					cv2.circle(flip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 438:
					location = "center"
					cv2.circle(flip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 195:
					location = "left_top"
					cv2.circle(flip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 379:
					location = "center_top"
					cv2.circle(flip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 316:
					location = "right_bottom"
					cv2.circle(flip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)					

				elif index == 455:
					location = "center_right"
					cv2.circle(flip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

				elif index == 431:		
					location = "center_left"
					cv2.circle(flip_image, vis_obj, 1, (255, 0, 255), 4)

					if 0 < int(obj_point[0][0]) < 640 and 0 < int(obj_point[0][1]) < 480:
						zc = cv_depth[int(obj_point[0][1]), int(obj_point[0][0])]/1000
						obj_rx, obj_ry, obj_rz = self.getXYZ(obj_point[0][0] /1.0 , obj_point[0][1], zc/1.0 )
						position_msgs = self.append_to_array(position_msgs, obj_rx, obj_ry, obj_rz, location)

	# ================= Publisher ==================
			flip_fig.canvas.draw()
			w, h = flip_fig.canvas.get_width_height()
			predict_flip = np.fromstring(flip_fig.canvas.tostring_argb(), dtype=np.uint8)
			predict_flip.shape = (w, h, 4)

			converted_img = cv2.cvtColor(predict_flip, cv2.COLOR_BGRA2BGR)
			position_msgs.image = self.bridge.cv2_to_imgmsg(converted_img, "8UC3")
			self.predict_right_3D.publish(self.bridge.cv2_to_imgmsg(predict_flip, "8UC4"))
			self.pose_pub.publish(position_msgs)
			# Flip
			predict_flip = cv2.flip(flip_image, 1)
			self.predict_right_2D.publish(self.bridge.cv2_to_imgmsg(predict_flip, "bgr8"))
		
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

	def forward_pass_3d(self, input_image, pred_obj=True):

		sample = {}
		sample[TransQueries.images] = input_image
		sample[BaseQueries.sides] = ["left"]
		sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
		sample["root"] = "wrist"
		if pred_obj:
			sample[TransQueries.objpoints3d] = input_image.new_ones((1, 600, 3)).float()
		_, results, _ = self.model_obman.forward(sample, no_loss=True)

		return results

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

	def switch_callback(self, req):
		resp = obman_switchResponse()
		self.switch = req.data
		s = "True" if req.data else "False"
		resp.result = "Switch turn to {}".format(req.data)
		return resp

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	
	

if __name__ == '__main__': 
	rospy.init_node('obman_prediction',anonymous=False)
	obman_prediction = obman_prediction()
	rospy.on_shutdown(obman_prediction.onShutdown)
	rospy.spin()
