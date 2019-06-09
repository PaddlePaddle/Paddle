cmake_minimum_required(VERSION 3.0)

project(cpp_train_demo CXX C)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/paddle/lib/dir")
endif()

option(WITH_MKLDNN     "Compile PaddlePaddle with MKLDNN"                                   OFF)
option(WITH_MKL        "Compile PaddlePaddle with MKL support, default use openblas."       OFF)

include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/snappy/include")
include_directories("${PADDLE_LIB}/third_party/install/snappystream/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")

include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

link_directories("${PADDLE_LIB}/third_party/install/snappy/lib")
link_directories("${PADDLE_LIB}/third_party/install/snappystream/lib")
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")

add_executable(demo_trainer demo_trainer.cc)

if(WITH_MKLDNN)
  include_directories("${PADDLE_LIB}/third_party/install/mkldnn/include")
  if(WIN32)
    set(MKLDNN_LIB ${PADDLE_LIB}/third_party/install/mkldnn/lib/mkldnn.lib)
  else(WIN32)
    set(MKLDNN_LIB ${PADDLE_LIB}/third_party/install/mkldnn/lib/libmkldnn.so.0)
  endif(WIN32)
endif(WITH_MKLDNN)

if(WITH_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
  if(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.lib)
  else(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel.so)
  endif(WIN32)
else()
  if(APPLE)
    set(MATH_LIB cblas)
  elseif(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas.lib)
  else()
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas.a)
  endif(APPLE)
endif()

if(APPLE)
  set(MACOS_LD_FLAGS "-undefined dynamic_lookup -Wl,-all_load -framework CoreFoundation -framework Security")
else(APPLE)
  set(ARCHIVE_START "-Wl,--whole-archive")
  set(ARCHIVE_END "-Wl,--no-whole-archive")
  set(EXTERNAL_LIB "-lrt -ldl -lpthread")
endif(APPLE)

target_link_libraries(demo_trainer
        ${MACOS_LD_FLAGS}
        ${ARCHIVE_START}
        ${PADDLE_LIB}/paddle/fluid/inference/libpaddle_fluid.a
        ${ARCHIVE_END}
        ${MATH_LIB}
        ${MKLDNN_LIB}
        glog gflags protobuf snappystream snappy z xxhash
        ${EXTERNAL_LIB})
