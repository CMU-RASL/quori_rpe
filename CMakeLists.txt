cmake_minimum_required(VERSION 2.8.3)
project(openface2_ros)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  image_transport
  cv_bridge
  std_msgs
  geometry_msgs
  sensor_msgs
  roscpp
  tf2
  tf2_ros
  rosbag
)

find_package(Boost REQUIRED COMPONENTS filesystem system)

find_package(OpenCV REQUIRED)

find_library(OPENFACE_LD_LIB LandmarkDetector)
find_library(OPENFACE_FA_LIB FaceAnalyser)
find_library(OPENFACE_GA_LIB GazeAnalyser)
find_library(OPENFACE_UT_LIB Utilities)

set(OPENFACE_LIBS ${OPENFACE_LD_LIB} ${OPENFACE_FA_LIB} ${OPENFACE_GA_LIB} ${OPENFACE_UT_LIB})

find_path(OPENFACE_INCLUDE_DIR LandmarkDetectorModel.h PATH_SUFFIXES OpenFace)

add_message_files(FILES ActionUnit.msg Face.msg Faces.msg)
generate_messages(DEPENDENCIES std_msgs geometry_msgs)

# add_definitions(-g -pg)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OPENFACE_INCLUDE_DIR}
  ${OpenBLAS_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}/boost
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(quori_openface2_ros src/quori_openface2_ros.cpp)
add_dependencies(quori_openface2_ros ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(quori_openface2_ros openface2_ros_generate_messages_cpp)

add_executable(quori_openface2_rosbag src/quori_openface2_rosbag.cpp)
add_dependencies(quori_openface2_rosbag ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(quori_openface2_rosbag openface2_ros_generate_messages_cpp)


install(TARGETS quori_openface2_ros
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

target_link_libraries(quori_openface2_ros
  ${catkin_LIBRARIES}
  ${OPENFACE_LIBS}
  tbb
  openblas
  dlib
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Python_LIBRARIES}
)

install(TARGETS quori_openface2_rosbag
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

target_link_libraries(quori_openface2_rosbag
  ${catkin_LIBRARIES}
  ${OPENFACE_LIBS}
  tbb
  openblas
  dlib
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Python_LIBRARIES}
)
