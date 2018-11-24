cmake_minimum_required(VERSION 3.0)
project(cpp_inference_demo CXX C)
option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(USE_TENSORRT "Compile demo with TensorRT."   OFF)

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if (WIN32)
  if (WITH_STATIC_LIB)
    safe_set_static_flag()
    add_definitions(-DSTATIC_LIB)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "/w")
    set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "/w")
  endif()
  set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  set(CMAKE_STATIC_LIBRARY_PREFIX "")
endif()
message("flags" ${CMAKE_CXX_FLAGS})

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()


if(WITH_GPU)
  if(NOT WIN32)
    set(CUDA_LIB "/usr/local/cuda/lib64/" CACHE STRING "CUDA Library")
  else()
    if(CUDA_LIB STREQUAL "")
    set(CUDA_LIB "C:\\Program\ Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64")
    endif()
  endif(NOT WIN32)
endif()
include_directories("${PADDLE_LIB}")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
if (NOT WIN32)
include_directories("${PADDLE_LIB}/third_party/install/snappy/include")
include_directories("${PADDLE_LIB}/third_party/install/snappystream/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
endif(NOT WIN32)

include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

if (NOT WIN32)
  if (USE_TENSORRT AND WITH_GPU)
      include_directories("${TENSORRT_INCLUDE_DIR}")
      link_directories("${TENSORRT_LIB_DIR}")
  endif()
endif(NOT WIN32)

if (NOT WIN32)
link_directories("${PADDLE_LIB}/third_party/install/snappy/lib")
link_directories("${PADDLE_LIB}/third_party/install/snappystream/lib")
link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
endif(NOT WIN32)

link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

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
      ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(DEPS
      ${PADDLE_LIB}/paddle/lib/libpaddle_fluid${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

if (NOT WIN32)
set(EXTERNAL_LIB "-lrt -ldl -lpthread")
set(DEPS ${DEPS}
    ${MATH_LIB} ${MKLDNN_LIB}
    glog gflags protobuf snappystream snappy z xxhash
    ${EXTERNAL_LIB})
else()
set(DEPS ${DEPS}
    ${MATH_LIB} ${MKLDNN_LIB}
    ${CMAKE_STATIC_LIBRARY_PREFIX}glog  ${CMAKE_STATIC_LIBRARY_PREFIX}gflags  ${CMAKE_STATIC_LIBRARY_PREFIX}protobuf
    ${EXTERNAL_LIB})
# NOTE(dzhwinter) shlwapi is deprecated.
set(DEPS ${DEPS} libcmt shlwapi)
endif(NOT WIN32)

if(WITH_GPU)
  if(NOT WIN32)
    if (USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
  set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
  set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX} )
  endif()
endif()

target_link_libraries(${DEMO_NAME} ${DEPS})
