#!/usr/bin/env python3

import numpy as np
import heapq
import time

# ROS
import rospy
import roslib
import rospkg
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Trigger, TriggerRequest
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from medical_msgs.msg import *


class arm_track_hand(object):
	def __init__(self):

		self.bridge = CvBridge()
		
		# Subscriber
		rospy.Subscriber('obman/hand_object_pose', HandObjectPoseArray, self.callback, queue_size = 1)

		# Publisher
		self.grip_pos = rospy.Publisher("mano/grip_pose", HandObjectPose, queue_size=1)
			
	def callback(self, msg):

		self.x = 0
		self.y = 0
		self.z = 0
		count = 0

		length = sum(np.array(msg.pose_array).shape)
		if int(length) == 30:
			for i in range(21):
				if msg.pose_array[i].pose.position.x > 0.3 and msg.pose_array[i].pose.position.x < 2:
					self.x += msg.pose_array[i].pose.position.x
					self.y += msg.pose_array[i].pose.position.y
					self.z += msg.pose_array[i].pose.position.z

					count += 1

		if count == 0:
			count = 1

		self.x = self.x / count
		self.y = self.y / count
		self.z = self.z / count

		grip_pos = HandObjectPose()
		if self.x != 0:
			grip_pos.pose.position.x = self.x
			grip_pos.pose.position.y = self.y
			grip_pos.pose.position.z = self.z
			grip_pos.pose.orientation.x = 0
			grip_pos.pose.orientation.y = 0
			grip_pos.pose.orientation.z = 0
			grip_pos.pose.orientation.w = 1

		self.grip_pos.publish(grip_pos)

		msg = None
				

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	
	

if __name__ == '__main__': 
	rospy.init_node('arm_track_hand',anonymous=False)
	arm_track_hand = arm_track_hand()
	rospy.on_shutdown(arm_track_hand.onShutdown)
	rospy.spin()
