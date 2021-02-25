#!/usr/bin/env python3

import numpy as np
import heapq
import time

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
from medical_msgs.srv import *



class grip_algorithm(object):
	def __init__(self):
		self.data_pose = HandObjectPose()
		self.old_predict_pose = Point()
		self.switch = False

		# Subscriber
		rospy.Subscriber("/obj_tf_pose", HandObjectPose, self.callback, queue_size=1)

		# Publisher
		self.target_pos = rospy.Publisher("/target_pose", HandObjectPose, queue_size=1)

		# service
		self.algorithm_switch = rospy.Service("~algorithm_switch_server", algorithm_switch, self.switch_callback)

		self.count = 0
		self.count_thres = 3
		self.dis_thres = 0.05
			
	def callback(self, msg):
		if not self.switch:
			return

		self.msg = msg
		real_world_point = msg.pose.position

		self.matching(real_world_point)
		self.msg = None

	def matching(self, real_world_point):
		lock = 1
		p_s = real_world_point
		p_t = self.old_predict_pose
		if np.sqrt((p_s.x - p_t.x)**2 + (p_s.y - p_t.y)**2 + (p_s.z - p_t.z)**2) < self.dis_thres:
			self.count = self.count + 1
			real_world_point.x = (p_s.x*self.count + p_t.x) / (self.count+1)
			real_world_point.y = (p_s.y*self.count + p_t.y) / (self.count+1)
			real_world_point.z = (p_s.z*self.count + p_t.z) / (self.count+1)	
		
		else:
			self.count = 0		

		self.old_predict_pose = real_world_point

		if self.count == self.count_thres:
			lock = 0

		if lock == 0:
			self.count = 0
			
			target_pose = HandObjectPose()
			target_pose.location = "center"
			target_pose.pose.position.x = real_world_point.x
			target_pose.pose.position.y = real_world_point.y
			target_pose.pose.position.z = real_world_point.z 
			target_pose.pose.orientation.x = self.msg.pose.orientation.x
			target_pose.pose.orientation.y = self.msg.pose.orientation.y
			target_pose.pose.orientation.z = self.msg.pose.orientation.z
			target_pose.pose.orientation.w = self.msg.pose.orientation.w

			self.target_pos.publish(target_pose)
			print(target_pose)
			time.sleep(0.5)

			try:
				do_grasping = rospy.ServiceProxy('handover_pick', Trigger) # call service
				resp = do_grasping()

			except rospy.ServiceException as exc:
				print("service did not process request: " + str(exc))

			time.sleep(5)

	def switch_callback(self, req):
		resp = obman_switchResponse()
		self.switch = req.data				

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	
	

if __name__ == '__main__': 
	rospy.init_node('grip_algorithm',anonymous=False)
	grip_algorithm = grip_algorithm()
	rospy.on_shutdown(grip_algorithm.onShutdown)
	rospy.spin()
