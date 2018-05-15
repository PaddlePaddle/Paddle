cmake_minimum_required(VERSION 3.0)

project(cpp_train_demo CXX C)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories("${PROJECT_SOURCE_DIR}/lib")

set(PADDLE_LIB "${PROJECT_SOURCE_DIR}/lib")

include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/snappy/include")
include_directories("${PADDLE_LIB}/third_party/install/snappystream/include")

include_directories("${PADDLE_LIB}/third_party//boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

link_directories("${PADDLE_LIB}/third_party/install/snappystream/lib")
link_directories("${PADDLE_LIB}/third_party/install/snappy/lib")
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")

add_executable(demo_trainer demo_trainer.cc)

target_link_libraries(demo_trainer ${PADDLE_LIB}/paddle/fluid/inference/libpaddle_fluid.a glog gflags protobuf snappy snappystream z)
