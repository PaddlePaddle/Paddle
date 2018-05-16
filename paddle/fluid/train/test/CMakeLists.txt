cmake_minimum_required(VERSION 3.0)

project(cpp_train_demo CXX C)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(PADDLE_LIB "${PROJECT_SOURCE_DIR}/lib")

include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/snappy/include")
include_directories("${PADDLE_LIB}/third_party/install/snappystream/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")

include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

link_directories("${PADDLE_LIB}/third_party/install/snappystream/lib")
link_directories("${PADDLE_LIB}/third_party/install/snappy/lib")
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")

add_executable(demo_trainer demo_trainer.cc)

IF(APPLE)
  SET(MACOS_LD_FLAGS "-undefined dynamic_lookup -Wl,-all_load -framework CoreFoundation -framework Security")
ELSE(APPLE)
  SET(ARCHIVE_START "-Wl,--whole-archive")
  SET(ARCHIVE_END "-Wl,--no-whole-archive")
ENDIF(APPLE)

target_link_libraries(demo_trainer
        ${MACOS_LD_FLAGS}
        ${ARCHIVE_START}
        ${PADDLE_LIB}/paddle/fluid/inference/libpaddle_fluid.a
        ${ARCHIVE_END}
        glog gflags protobuf snappy snappystream z cblas)
