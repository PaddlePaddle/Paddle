cmake_minimum_required(VERSION 3.0)

project(cpp_inference_demo CXX C)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if (WIN32)
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
else()
set(CMAKE_STATIC_LIBRARY_PREFIX "")
endif()

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)

if(WITH_GPU)
  set(CUDA_LIB "/usr/local/cuda/lib64/" CACHE STRING "CUDA Library")
endif()

include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
if (NOT WIN32)
include_directories("${PADDLE_LIB}/third_party/install/snappy/include")
include_directories("${PADDLE_LIB}/third_party/install/snappystream/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
endif(NOT WIN32)

include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

if (NOT WIN32)
link_directories("${PADDLE_LIB}/third_party/install/snappy/lib")
link_directories("${PADDLE_LIB}/third_party/install/snappystream/lib")
link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
endif(NOT WIN32)

link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/paddle/fluid/inference")

add_executable(${DEMO_NAME} ${DEMO_NAME}.cc)

if(WITH_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX} 
               ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
  endif()
else()
  set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

# Note: libpaddle_inference_api.so/a must put before libpaddle_fluid.so/a
if(WITH_STATIC_LIB)
  set(DEPS
      ${PADDLE_LIB}/paddle/fluid/inference/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(DEPS
      ${PADDLE_LIB}/paddle/fluid/inference/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

if (NOT WIN32)
set(EXTERNAL_LIB "-lrt -ldl -lpthread")
set(DEPS ${DEPS}
    ${MATH_LIB} ${MKLDNN_LIB}
    glog gflags protobuf snappystream snappy z
    ${EXTERNAL_LIB})
else()
set(DEPS ${DEPS}
    ${MATH_LIB} ${MKLDNN_LIB}
    ${CMAKE_STATIC_LIBRARY_PREFIX}glog  ${CMAKE_STATIC_LIBRARY_PREFIX}gflags  ${CMAKE_STATIC_LIBRARY_PREFIX}protobuf
    ${EXTERNAL_LIB})
endif(NOT WIN32)

if(WITH_GPU)
  set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

target_link_libraries(${DEMO_NAME} ${DEPS})
