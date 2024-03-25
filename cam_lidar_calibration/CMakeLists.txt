cmake_minimum_required(VERSION 2.8.3)
project(cam_lidar_calibration)
add_definitions(-w)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")




find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  dynamic_reconfigure
  eigen_conversions
  image_transport
  geometry_msgs
  message_generation
  nodelet
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  rviz
  sensor_msgs
  std_msgs
  )

#set(OpenCV_DIR /home/lin/software/opencv-3.2_install/share/OpenCV)
find_package(OpenCV 3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(Qt5 REQUIRED Core Widgets)

## Setup include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  )

link_directories(${catkin_LIBRARY_DIRS})

add_definitions(-DQT_NO_KEYWORDS)

set(CMAKE_AUTOMOC ON)

add_action_files(
  FILES
  RunOptimise.action
  )
add_service_files(
  FILES
  Optimise.srv
  )
generate_messages(
  DEPENDENCIES
  actionlib_msgs
  geometry_msgs
  std_msgs
  sensor_msgs
  )
generate_dynamic_reconfigure_options(
  cfg/bounds.cfg
  )
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS nodelet
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_runtime
#    libutransform
  )

add_library(cam_lidar_calibration
  src/cam_lidar_panel.cpp
  src/feature_extractor.cpp
  src/load_params.cpp
  src/optimiser.cpp
        )
target_link_libraries(cam_lidar_calibration
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Eigen_LIBRARIES}
  Qt5::Widgets
  )

add_executable(feature_extraction_node src/feature_extraction_node.cpp)
target_link_libraries(feature_extraction_node
        cam_lidar_calibration
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${Eigen_LIBRARIES}
        )

add_executable(assess_node src/assess_calibration.cpp)
target_link_libraries(assess_node
        cam_lidar_calibration
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        )

#############
## Install ##
#############

## Mark the nodelet library for installations
install(TARGETS
  feature_extraction_node
  cam_lidar_calibration
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES nodelet_definition.xml plugin_description.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  )

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  )

install(DIRECTORY cfg/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cfg
  )

install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
  )

catkin_install_python(PROGRAMS 
  scripts/visualise_results.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)