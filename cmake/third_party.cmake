# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.

# compile third party for fluid on both windows/linux/mac

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
  "A path setting third party libraries download & build directories.")

set(THIRD_PARTY_BUILD_TYPE Release)

# Correction of flags on different Platform(WIN/MAC) and Print Warning Message
if (APPLE)
    if(WITH_MKL)
        MESSAGE(WARNING 
            "Mac is not supported with MKL in Paddle yet. Force WITH_MKL=OFF.")
        set(WITH_MKL OFF CACHE STRING "Disable MKL for building on mac" FORCE)
    endif()
endif()

if(WIN32 OR APPLE)
    MESSAGE(STATUS "Disable XBYAK in Windows and MacOS")
    SET(WITH_XBYAK OFF CACHE STRING "Disable XBYAK in Windows and MacOS" FORCE)

    if(WITH_LIBXSMM)
        MESSAGE(WARNING 
            "Windows, Mac are not supported with libxsmm in Paddle yet."
            "Force WITH_LIBXSMM=OFF")
        SET(WITH_LIBXSMM OFF CACHE STRING "Disable LIBXSMM in Windows and MacOS" FORCE)
    endif()

    if(WITH_NGRAPH)
        MESSAGE(WARNING
            "Windows or Mac is not supported with nGraph in Paddle yet."
            "Force WITH_NGRAPH=OFF")
        SET(WITH_NGRAPH OFF CACHE STRING "Disable nGraph in Windows and MacOS" FORCE)
    endif()

    if(WITH_BOX_PS)
        MESSAGE(WARNING
            "Windows or Mac is not supported with BOX_PS in Paddle yet."
            "Force WITH_BOX_PS=OFF")
        SET(WITH_BOX_PS OFF CACHE STRING "Disable BOX_PS package in Windows and MacOS" FORCE)
    endif()

    if(WITH_PSLIB)
        MESSAGE(WARNING
            "Windows or Mac is not supported with PSLIB in Paddle yet."
            "Force WITH_PSLIB=OFF")
        SET(WITH_PSLIB OFF CACHE STRING "Disable PSLIB package in Windows and MacOS" FORCE)
    endif()

    if(WITH_LIBMCT)
        MESSAGE(WARNING
            "Windows or Mac is not supported with LIBMCT in Paddle yet."
            "Force WITH_LIBMCT=OFF")
        SET(WITH_LIBMCT OFF CACHE STRING "Disable LIBMCT package in Windows and MacOS" FORCE)
    endif()

    if(WITH_PSLIB_BRPC)
        MESSAGE(WARNING
            "Windows or Mac is not supported with PSLIB_BRPC in Paddle yet."
            "Force WITH_PSLIB_BRPC=OFF")
        SET(WITH_PSLIB_BRPC OFF CACHE STRING "Disable PSLIB_BRPC package in Windows and MacOS" FORCE)
    endif()
endif()

set(WITH_MKLML ${WITH_MKL})
if(NOT DEFINED WITH_MKLDNN)
    if(WITH_MKL AND AVX2_FOUND)
        set(WITH_MKLDNN ON)
    else()
        message(STATUS "Do not have AVX2 intrinsics and disabled MKL-DNN")
        set(WITH_MKLDNN OFF)
    endif()
endif()

if(WIN32 OR APPLE OR NOT WITH_GPU OR ON_INFER)
    set(WITH_DGC OFF)
endif()

if(${CMAKE_VERSION} VERSION_GREATER "3.5.2")
    set(SHALLOW_CLONE "GIT_SHALLOW TRUE") # adds --depth=1 arg to git clone of External_Projects
endif()

########################### include third_party accoring to flags ###############################
include(external/zlib)      # download, build, install zlib
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/boost)     # download boost
include(external/eigen)     # download eigen3
include(external/threadpool)# download threadpool
include(external/dlpack)    # download dlpack
include(external/xxhash)    # download, build, install xxhash
include(external/warpctc)   # download, build, install warpctc

set(third_party_deps)
list(APPEND third_party_deps extern_eigen3 extern_gflags extern_glog extern_boost extern_xxhash)
list(APPEND third_party_deps extern_zlib extern_dlpack extern_warpctc extern_threadpool)

if(WITH_AMD_GPU)
    include(external/rocprim)   # download, build, install rocprim
    list(APPEND third_party_deps extern_rocprim)
endif()

include(cblas)              	# find first, then download, build, install openblas
if(${CBLAS_PROVIDER} STREQUAL MKLML)
    list(APPEND third_party_deps extern_mklml)
endif()
if(${CBLAS_PROVIDER} STREQUAL EXTERN_OPENBLAS)
    list(APPEND third_party_deps extern_openblas)
endif()

if(WITH_MKLDNN)
    include(external/mkldnn)    # download, build, install mkldnn
    list(APPEND third_party_deps extern_mkldnn)
endif()

include(external/protobuf)  	# find first, then download, build, install protobuf
if(NOT PROTOBUF_FOUND OR WIN32)
    list(APPEND third_party_deps extern_protobuf)
endif()

if(WITH_PYTHON)
    include(external/python)    # find python and python_module
    include(external/pybind11)  # download pybind11
    list(APPEND third_party_deps extern_pybind)
endif()

IF(WITH_TESTING OR (WITH_DISTRIBUTE AND NOT WITH_GRPC))
    include(external/gtest)     # download, build, install gtest
    list(APPEND third_party_deps extern_gtest)
ENDIF()

if(WITH_GPU)
    include(external/cub)       # download cub
    list(APPEND third_party_deps extern_cub)
endif(WITH_GPU)

if(WITH_PSLIB)
    include(external/pslib)          # download, build, install pslib
    list(APPEND third_party_deps extern_pslib)
    if(WITH_LIBMCT)
        include(external/libmct)     # download, build, install libmct
        list(APPEND third_party_deps extern_libxsmm)
    endif()
    if(WITH_PSLIB_BRPC)
        include(external/pslib_brpc) # download, build, install pslib_brpc
        list(APPEND third_party_deps extern_pslib_brpc)
    endif()
endif(WITH_PSLIB)

if(WITH_BOX_PS)
    include(external/box_ps)
    list(APPEND third_party_deps extern_box_ps)
endif(WITH_BOX_PS)

if(WITH_DISTRIBUTE)
    list(APPEND third_party_deps extern_cares)
    if(WITH_GRPC)
        list(APPEND third_party_deps extern_grpc)
    else()
        list(APPEND third_party_deps extern_leveldb)
        list(APPEND third_party_deps extern_brpc)
    endif()
endif()

if(WITH_NGRAPH)
    if(WITH_MKLDNN)
        include(external/ngraph)    # download, build, install nGraph
        list(APPEND third_party_deps extern_ngraph)
    else()
        MESSAGE(WARNING
            "nGraph needs mkl-dnn to be enabled."
            "Force WITH_NGRAPH=OFF")
        SET(WITH_NGRAPH OFF CACHE STRING "Disable nGraph if mkl-dnn is disabled" FORCE)
    endif()
endif()

if(WITH_XBYAK)
    include(external/xbyak)         # download, build, install xbyak
    list(APPEND third_party_deps extern_xbyak)
endif()

if(WITH_LIBXSMM)
    include(external/libxsmm)       # download, build, install libxsmm
    list(APPEND third_party_deps extern_libxsmm)
endif()

if(WITH_DGC)
    message(STATUS "add dgc lib.")
    include(external/dgc)           # download, build, install dgc
    add_definitions(-DPADDLE_WITH_DGC)
    list(APPEND third_party_deps extern_dgc)
endif()

add_custom_target(third_party DEPENDS ${third_party_deps})
