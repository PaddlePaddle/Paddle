# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PADDLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PADDLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(system)

project(paddle CXX C)
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

if(WIN32)
    set(CMAKE_SUPPRESS_REGENERATION ON)
    set(CMAKE_STATIC_LIBRARY_PREFIX lib)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    add_compile_options(/wd4068 /wd4129 /wd4244 /wd4267 /wd4297 /wd4530 /wd4577 /wd4819 /wd4838)
    set(PADDLE_LINK_FLAGS "/IGNORE:4006 /IGNORE:4098 /IGNORE:4217 /IGNORE:4221")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
endif(WIN32)

find_package(CUDA QUIET)
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

include(simd)

################################ Configurations #######################################
option(WITH_GPU         "Compile PaddlePaddle with NVIDIA GPU"          ${CUDA_FOUND})
option(WITH_AMD_GPU     "Compile PaddlePaddle with AMD GPU"             OFF)
option(WITH_AVX         "Compile PaddlePaddle with AVX intrinsics"      ${AVX_FOUND})
option(WITH_MKL         "Compile PaddlePaddle with MKL support."        ${AVX_FOUND})
option(WITH_NGRAPH      "Compile PaddlePaddle with nGraph support."     OFF)
option(WITH_DSO         "Compile PaddlePaddle with dynamic linked CUDA" ON)
option(WITH_TESTING     "Compile PaddlePaddle with unit testing"        OFF)
option(WITH_PYTHON      "Compile PaddlePaddle with python interpreter"  ON)
option(WITH_PROFILER    "Compile PaddlePaddle with GPU profiler and gperftools"        OFF)
option(WITH_JEMALLOC    "Compile PaddlePaddle with jemalloc"            OFF)
option(WITH_COVERAGE    "Compile PaddlePaddle with code coverage"       OFF)
option(COVERALLS_UPLOAD "Package code coverage data to coveralls"       OFF)
option(WITH_DISTRIBUTE  "Compile with distributed support"              OFF)
option(WITH_PSLIB       "Compile with pslib support"                    OFF)
option(WITH_CONTRIB     "Compile the third-party contributation"        OFF)
option(REPLACE_ENFORCE_GLOG "Replace PADDLE_ENFORCE with glog/CHECK for better debug." OFF)
option(WITH_ANAKIN      "Compile with Anakin library"                   OFF)
option(ANAKIN_BUILD_FAT_BIN "Build anakin cuda fat-bin lib for all device plantform, ignored when WITH_ANAKIN=OFF" OFF)
option(ANAKIN_BUILD_CROSS_PLANTFORM "Build anakin lib for any nvidia device plantform. ignored when WITH_ANAKIN=OFF" ON)
option(WITH_GRPC     "Use grpc as the default rpc framework"            ${WITH_DISTRIBUTE})
option(WITH_BRPC_RDMA     "Use brpc rdma as the rpc protocal"           OFF)
option(ON_INFER         "Turn on inference optimization."               OFF)
option(WITH_INFERENCE_API_TEST   "Test fluid inference high-level api interface"  OFF)
option(WITH_SYSTEM_BLAS   "Use system blas library"           OFF)
option(PY_VERSION       "Compile PaddlePaddle with python3 support"     ${PY_VERSION})
option(WITH_FAST_MATH   "Make use of fast math library, might affect the precision to some extent" ON)

# PY_VERSION
if(NOT PY_VERSION)
  set(PY_VERSION 2.7)
endif()
set(PYBIND11_PYTHON_VERSION ${PY_VERSION})

# CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
endif()

if (APPLE)
    set(WITH_MKL OFF CACHE STRING
        "Disable MKL for building on mac" FORCE)
endif()

if (WIN32)
    set(WITH_DISTRIBUTE OFF CACHE STRING
            "Disable DISTRIBUTE when compiling for Windows" FORCE)
endif()

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
  "A path setting third party libraries download & build directories.")

set(FLUID_INSTALL_DIR "${CMAKE_BINARY_DIR}/fluid_install_dir" CACHE STRING
  "A path setting fluid shared and static libraries")

set(FLUID_INFERENCE_INSTALL_DIR "${CMAKE_BINARY_DIR}/fluid_inference_install_dir" CACHE STRING
  "A path setting fluid inference shared and static libraries")

set(THIRD_PARTY_BUILD_TYPE Release)

set(WITH_MKLML ${WITH_MKL})
if (NOT DEFINED WITH_MKLDNN)
    if (WITH_MKL AND AVX2_FOUND)
        set(WITH_MKLDNN ON)
    else()
        message(STATUS "Do not have AVX2 intrinsics and disabled MKL-DNN")
        set(WITH_MKLDNN OFF)
    endif()
endif()

if (REPLACE_ENFORCE_GLOG)
  add_definitions("-DREPLACE_ENFORCE_GLOG")
endif()
########################################################################################

include(external/mklml)     # download mklml package
include(external/xbyak)     # download xbyak package
include(external/libxsmm)   # download, build, install libxsmm
include(external/zlib)      # download, build, install zlib
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/gtest)     # download, build, install gtest
include(external/protobuf)  # download, build, install protobuf
include(external/python)    # download, build, install python
include(external/openblas)  # download, build, install openblas
include(external/mkldnn)    # download, build, install mkldnn
include(external/ngraph)    # download, build, install nGraph
include(external/boost)     # download boost
include(external/eigen)     # download eigen3
include(external/pybind11)  # download pybind11
include(external/cares)
include(external/cub)
include(external/rocprim)
include(external/xxhash)    # download xxhash
include(external/dlpack)
include(external/snappy)    # download snappy
include(external/snappystream) # download snappystream
include(external/warpctc)   # download, build, install warpctc
include(external/wbaes)   # download wbaes

if (NOT WIN32)
# there is no official support of nccl, cupti in windows
include(cupti)
include(external/gzstream)
endif (NOT WIN32)

if(WITH_PSLIB)
    include(external/libmct)
    include(external/pslib_brpc)
    include(external/pslib)
endif(WITH_PSLIB)

if(WITH_DISTRIBUTE)
    if(WITH_GRPC)
        include(external/grpc)
        message(STATUS "Use grpc framework.")
    else()
        message(STATUS "Use brpc framework.")
        include(external/leveldb)
        include(external/brpc)
    endif()
endif()

if(WITH_BRPC_RDMA)
    message(STATUS "Use brpc with rdma.")
    if(WITH_GRPC)
        message(FATAL_ERROR "Can't use grpc with brpc rdma.")
    endif()
    if(NOT WITH_DISTRIBUTE)
        message(FATAL_ERROR "Can't use brpc rdma in no distribute env.")
    endif()
endif()


include(external/threadpool)
include(flags)              # set paddle compile flags
include(cudnn)              # set cudnn libraries, must before configure
include(configure)          # add paddle env configuration

if(WITH_GPU)
    include(cuda)
    include(tensorrt)
endif()
if(WITH_MKL OR WITH_MKLML)
    include(external/anakin)
elseif()
    set(WITH_ANAKIN OFF CACHE STRING "Anakin is used in MKL only now." FORCE)
endif()

if (WITH_PROFILER)
    find_package(Gperftools REQUIRED)
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
    add_definitions(-DWITH_GPERFTOOLS)
endif()

if (WITH_JEMALLOC)
    find_package(JeMalloc REQUIRED)
    include_directories(${JEMALLOC_INCLUDE_DIR})
    add_definitions(-DPADDLE_WITH_JEMALLOC)
endif()

include(generic)            # simplify cmake module
include(package)            # set paddle packages
include(ccache)             # set ccache for compilation
include(util)               # set unittest and link libs
include(version)            # set PADDLE_VERSION
include(coveralls)          # set code coverage
include(inference_lib)      # add paddle fluid inference libraries


include_directories("${PADDLE_SOURCE_DIR}")

if(WITH_AMD_GPU)
    find_package(HIP)
    include(hip)
endif(WITH_AMD_GPU)

set(PADDLE_PYTHON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/build")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

if (ON_INFER)
    message(STATUS "On inference mode, will take place some specific optimization.")
    add_definitions(-DPADDLE_ON_INFERENCE)
else()
    #TODO(luotao), combine this warning with `make inference_lib_dist` command.
    message(WARNING "On inference mode, will take place some specific optimization. Turn on the ON_INFER flag when building inference_lib only.")
endif()

add_subdirectory(paddle)
if(WITH_PYTHON)
    add_subdirectory(python)
endif()
