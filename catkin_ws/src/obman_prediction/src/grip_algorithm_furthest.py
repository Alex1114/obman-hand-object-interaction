#!/usr/bin/env python3

import numpy as np
import heapq
import time
import math

# ROS
import rospy
import roslib
import rospkg
from std_srvs.srv import Trigger, TriggerRequest
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from medical_msgs.msg import *


class grip_algorithm(object):
	def __init__(self):
		self.data_pose = HandObjectPose()
		self.old_predict_pose = Point()

		# Subscriber
		rospy.Subscriber("/obman/grip_pose", HandObjectPoseArray, self.callback, queue_size=1)

		# Publisher
		self.target_pos = rospy.Publisher("/target_pose", HandObjectPose, queue_size=1)

		self.count = 0
		self.count_thres = 5
		self.dis_thres = 0.05
			
	def callback(self, msg):
		target = self.target_search(msg)

		real_world_point = target.pose

		self.matching(real_world_point)

		self.msg = None
	
	def target_search(self, msg):
		target = HandObjectPose()
		temp_target_pose_array = HandObjectPoseArray()
		
		self.x = 0
		self.y = 0
		self.z = 0
		distance = []
		count = 0
		length = sum(np.array(msg.pose_array).shape)
		for i in range(length):
			if msg.pose_array[i].location == "finger" and msg.pose_array[i].pose.position.x >= 0.2 and msg.pose_array[i].pose.position.x <= 1.2:
				self.x += msg.pose_array[i].pose.position.x
				self.y += msg.pose_array[i].pose.position.y
				self.z += msg.pose_array[i].pose.position.z
				count += 1

		if count == 0:
			count = 1

		self.x = self.x / count
		self.y = self.y / count
		self.z = self.z / count

		for i in range(length):
			if msg.pose_array[i].location != "finger" and msg.pose_array[i].pose.position.x >= 0.2 and msg.pose_array[i].pose.position.x <= 1.2:
				dis = math.sqrt((msg.pose_array[i].pose.position.x - self.x) ** 2 + (msg.pose_array[i].pose.position.y - self.y) ** 2 + (msg.pose_array[i].pose.position.z - self.z) ** 2)
				distance.append(dis)
				
				temp_target_pose = HandObjectPose()
				temp_target_pose.location = msg.pose_array[i].location
				temp_target_pose.pose.position.x = msg.pose_array[i].pose.position.x
				temp_target_pose.pose.position.y = msg.pose_array[i].pose.position.y
				temp_target_pose.pose.position.z = msg.pose_array[i].pose.position.z
				temp_target_pose.pose.orientation.x = msg.pose_array[i].pose.orientation.x
				temp_target_pose.pose.orientation.y = msg.pose_array[i].pose.orientation.y
				temp_target_pose.pose.orientation.z = msg.pose_array[i].pose.orientation.z
				temp_target_pose.pose.orientation.w = msg.pose_array[i].pose.orientation.w
				temp_target_pose_array.pose_array.append(temp_target_pose)

		target_index = distance.index(max(distance))
		
		target = HandObjectPose()
		target.location = temp_target_pose_array.pose_array[target_index].location
		target.pose.position.x = temp_target_pose_array.pose_array[target_index].pose.position.x
		target.pose.position.y = temp_target_pose_array.pose_array[target_index].pose.position.y
		target.pose.position.z = temp_target_pose_array.pose_array[target_index].pose.position.z
		target.pose.orientation.x = temp_target_pose_array.pose_array[target_index].pose.orientation.x
		target.pose.orientation.y = temp_target_pose_array.pose_array[target_index].pose.orientation.y
		target.pose.orientation.z = temp_target_pose_array.pose_array[target_index].pose.orientation.z
		target.pose.orientation.w = temp_target_pose_array.pose_array[target_index].pose.orientation.w
		return target


	def matching(self, real_world_point):
		
		lock = 1
		p_s = real_world_point.position
		p_t = self.old_predict_pose
		if np.sqrt((p_s.x - p_t.x)**2 + (p_s.y - p_t.y)**2 + (p_s.z - p_t.z)**2) < self.dis_thres:
			self.count = self.count + 1
			real_world_point.position.x = (p_s.x*self.count + p_t.x) / (self.count+1)
			real_world_point.position.y = (p_s.y*self.count + p_t.y) / (self.count+1)
			real_world_point.position.z = (p_s.z*self.count + p_t.z) / (self.count+1)	
		
		else:
			self.count = 0		

		self.old_predict_pose = real_world_point.position

		if self.count == self.count_thres:
			lock = 0

		if lock == 0:
			self.count = 0
			
			target_pose = HandObjectPose()
			target_pose.location = "target"
			target_pose.pose.position.x = real_world_point.position.x
			target_pose.pose.position.y = real_world_point.position.y
			target_pose.pose.position.z = real_world_point.position.z
			target_pose.pose.orientation.x = real_world_point.orientation.x
			target_pose.pose.orientation.y = real_world_point.orientation.y
			target_pose.pose.orientation.z = real_world_point.orientation.z
			target_pose.pose.orientation.w = real_world_point.orientation.w

			self.target_pos.publish(target_pose)
			print(target_pose)
				

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	
	

if __name__ == '__main__': 
	rospy.init_node('grip_algorithm',anonymous=False)
	grip_algorithm = grip_algorithm()
	rospy.on_shutdown(grip_algorithm.onShutdown)
	rospy.spin()