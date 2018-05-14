cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(cpp_train_demo CXX C)
add_executable(demo_trainer demo_trainer.cc)

include_directories("${PROJECT_SOURCE_DIR}/lib")
include_directories("${PROJECT_SOURCE_DIR}/lib/third_party/install/protobuf/include")
include_directories("${PROJECT_SOURCE_DIR}/lib/third_party/eigen3")
link_directories("${PROJECT_SOURCE_DIR}/lib/paddle/fluid/inference")
#link_libraries("/Users/qiaolongfei/project/paddle/paddle/fluid/train/test/lib/paddle/fluid/inference/libpaddle_fluid.a")
target_link_libraries(demo_trainer ${PROJECT_SOURCE_DIR}/lib/paddle/fluid/inference/libpaddle_fluid.a)
