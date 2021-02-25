catkin_make --pkg vision_opencv -C ./catkin_ws \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so

# touch ./catkin_ws/src/mm-core/bin_picking/ddqn/grasp_suck/CATKIN_IGNORE
# touch ./catkin_ws/src/mm-core/bin_picking/ddqn/visualization/CATKIN_IGNORE
# touch ./catkin_ws/src/mm-core/arm_operation/abb_control/CATKIN_IGNORE
# touch ./catkin_ws/src/subt-core/mapping_and_odometry/isam_sensors/CATKIN_IGNORE
# touch ./catkin_ws/src/subt-core/perception/image_roi_extraction/CATKIN_IGNORE
# touch ./catkin_ws/src/subt-core/perception/velodyne_perception/CATKIN_IGNORE


# catkin_make --pkg apriltags2_ros -C ./catkin_ws
# catkin_make --pkg realsense2_camera -C ./catkin_ws
# catkin_make --pkg cloud_msgs -C ./catkin_ws
# catkin_make --pkg subt_msgs -C ./catkin_ws
# catkin_make --pkg anchor_measure -C ./catkin_ws
# catkin_make --pkg sound_localize -C ./catkin_ws

catkin_make -C ./catkin_ws
