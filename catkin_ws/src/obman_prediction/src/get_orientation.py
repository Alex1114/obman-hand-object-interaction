#!/usr/bin/env python3

import os 
import cv2
import math
import time
import numpy as np
from PIL import Image
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
from scipy.spatial.transform import Rotation

class get_orientation(object):
	def __init__(self):
		self.bridge = CvBridge()
		
		# Subscriber
		rospy.Subscriber('obman/hand_object_pose', HandObjectPoseArray, self.callback, queue_size = 1)
		
		# Publisher
		self.predict_object = rospy.Publisher("obman/object_orientation", Image, queue_size = 1)
		self.grip_pose_pub = rospy.Publisher("obman/grip_pose", HandObjectPoseArray, queue_size=1)

	def callback(self, data):
		# Ros image to cv2
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data.image, "8UC3")
		except CvBridgeError as e:
			print(e)

		# Find contours
		gs_frame = cv2.GaussianBlur(cv_image, (5, 5), 0)                   
		hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)                 
		erode_hsv = cv2.erode(hsv, None, iterations=2)                 
		# inRange_hsv = cv2.inRange(erode_hsv, np.array([26, 43, 46]), np.array([34, 255, 255]))
		inRange_hsv = cv2.inRange(erode_hsv, np.array([100, 80, 46]), np.array([124, 255, 255]))

		cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		c = max(cnts, key=cv2.contourArea)
		rect = cv2.minAreaRect(c)
		box = cv2.boxPoints(rect)
		cv2.drawContours(cv_image, [np.int0(box)], -1, (0, 255, 255), 2)

		# Calculate angle
		angle = rect[-1]
		size1, size2 = rect[1][0], rect[1][1]
		ratio_size = float(size1) / float(size2)
		if 1.25 > ratio_size > 0.75:
			if angle < -45:
				angle = 90 + angle
		else:
			if size1 < size2:
				angle = angle + 180
			else:
				angle = angle + 90

			if angle > 90:
				angle = angle - 180

		# Create a rotation object from Euler angles specifying axes of rotation
		rot = Rotation.from_euler('xyz', [angle, 0, 0], degrees=True)

		# Convert to quaternions and print
		rot_quat = rot.as_quat()

		# Msgs
		grip_pose = HandObjectPoseArray()
		grip_pose.header = data.header
		grip_pose.pose_array = data.pose_array

		length = sum(np.array(grip_pose.pose_array).shape)
		if int(length) == 30:
			for i in range(21):
				grip_pose.pose_array[i].pose.orientation.x = 0
				grip_pose.pose_array[i].pose.orientation.y = 0
				grip_pose.pose_array[i].pose.orientation.z = 0
				grip_pose.pose_array[i].pose.orientation.w = 1
			
			for i in range(21,30):
				grip_pose.pose_array[i].pose.orientation.x = rot_quat[0]
				grip_pose.pose_array[i].pose.orientation.y = rot_quat[1]
				grip_pose.pose_array[i].pose.orientation.z = rot_quat[2]
				grip_pose.pose_array[i].pose.orientation.w = rot_quat[3]
			
		else:
			pass

		self.predict_object.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		self.grip_pose_pub.publish(grip_pose)
			



	def onShutdown(self):
		rospy.loginfo("Shutdown.")	

if __name__ == '__main__': 
	rospy.init_node('get_orientation',anonymous=False)
	get_orientation = get_orientation()
	rospy.on_shutdown(get_orientation.onShutdown)
	rospy.spin()

