# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

cmake_minimum_required(VERSION 3.0)

project(paddle CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(PROJ_ROOT ${CMAKE_SOURCE_DIR})

find_package(Sphinx)
find_package(CUDA QUIET)
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

include(system)
include(simd)

################################ Configurations #######################################
option(WITH_GPU         "Compile PaddlePaddle with NVIDIA GPU"          ${CUDA_FOUND})
option(WITH_AVX         "Compile PaddlePaddle with AVX intrinsics"      ${AVX_FOUND})
option(WITH_DSO         "Compile PaddlePaddle with dynamic linked CUDA" ON)
option(WITH_TESTING     "Compile PaddlePaddle with unit testing"        ON)
option(WITH_SWIG_PY     "Compile PaddlePaddle with inference api"       ON)
option(WITH_STYLE_CHECK "Compile PaddlePaddle with style check"         ON)
option(WITH_PYTHON      "Compile PaddlePaddle with python interpreter"  ON)
option(WITH_DOUBLE      "Compile PaddlePaddle with double precision"    OFF)
option(WITH_RDMA        "Compile PaddlePaddle with RDMA support"        OFF)
option(WITH_TIMER       "Compile PaddlePaddle with stats timer"         OFF)
option(WITH_PROFILER    "Compile PaddlePaddle with GPU profiler"        OFF)
option(WITH_DOC         "Compile PaddlePaddle with documentation"       OFF)
option(ON_COVERALLS     "Compile PaddlePaddle with code coverage"       OFF)
option(COVERALLS_UPLOAD "Package code coverage data to coveralls"       OFF)
option(ON_TRAVIS        "Exclude special unit test on Travis CI"        OFF)

# CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
endif()

set(THIRD_PARTY_PATH "${PROJ_ROOT}/third_party" CACHE STRING
  "A path setting third party libraries download & build directories.")
########################################################################################

include(external/zlib)      # download, build, install zlib
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/gtest)     # download, build, install gtest
include(external/protobuf)  # download, build, install protobuf
include(external/python)    # download, build, install python
include(external/openblas)  # download, build, install openblas
include(external/swig)      # download, build, install swig
include(external/warpctc)   # download, build, install warpctc

include(package)            # set paddle packages
include(cpplint)            # set paddle c++ style
include(ccache)             # set ccache for compilation
include(util)               # set unittest and link libs
include(rdma)               # set rdma libraries
include(flags)              # set paddle compile flags
include(cudnn)              # set cudnn libraries
include(version)            # set PADDLE_VERSION
include(coveralls)          # set code coverage

include(configure)          # add paddle env configuration

include_directories("${PROJ_ROOT}")
include_directories("${PROJ_ROOT}/paddle/cuda/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/proto")

set(EXTERNAL_LIBS
    # have not include gtest here.
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${CBLAS_LIBRARIES}
    ${PROTOBUF_LIBRARY}
    ${ZLIB_LIBRARIES}
)

add_subdirectory(proto)
add_subdirectory(paddle)
add_subdirectory(python)

if(WITH_DOC)
    add_subdirectory(doc)
endif()
