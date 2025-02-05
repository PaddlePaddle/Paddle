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
# limitations under the License.

include(ExternalProject)

if(WITH_ROCM)
  add_definitions(-DWARPCTC_WITH_HIP)
endif()

set(WARPCTC_PREFIX_DIR ${THIRD_PARTY_PATH}/warpctc)
set(WARPCTC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/warpctc)
# in case of low internet speed
#set(WARPCTC_REPOSITORY  https://gitee.com/tianjianhe/warp-ctc.git)
set(WARPCTC_TAG bdc2b4550453e0ef2d3b5190f9c6103a84eff184)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/warpctc)
set(WARPCTC_PATCH_COMMAND "")
set(WARPCTC_CCBIN_OPTION "")
if(WIN32)
  set(WARPCTC_PATCH_CUDA_COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      ${PADDLE_SOURCE_DIR}/patches/warpctc/CMakeLists.txt.cuda.patch
      "<SOURCE_DIR>/")
else()
  set(WARPCTC_PATCH_CUDA_COMMAND
      git checkout -- . && git checkout ${WARPCTC_TAG} && patch -Nd
      ${SOURCE_DIR} <
      ${PADDLE_SOURCE_DIR}/patches/warpctc/CMakeLists.txt.cuda.patch)
endif()

if(NOT WIN32 AND WITH_GPU)
  if(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0 AND ${CMAKE_CXX_COMPILER_VERSION}
                                                  VERSION_GREATER 12.0)
    file(TO_NATIVE_PATH
         ${PADDLE_SOURCE_DIR}/patches/warpctc/CMakeLists.txt.patch native_src)
    set(WARPCTC_PATCH_COMMAND git checkout -- . && git checkout ${WARPCTC_TAG}
                              && patch -Nd ${SOURCE_DIR} < ${native_src} &&)
    set(WARPCTC_CCBIN_OPTION -DCCBIN_COMPILER=${CCBIN_COMPILER})
  endif()
endif()

if(WITH_ROCM)
  set(WARPCTC_PATHCH_ROCM_COMMAND
      patch -p1 <
      ${PADDLE_SOURCE_DIR}/patches/warpctc/CMakeLists.txt.rocm.patch && patch
      -p1 < ${PADDLE_SOURCE_DIR}/patches/warpctc/devicetypes.cuh.patch && patch
      -p1 < ${PADDLE_SOURCE_DIR}/patches/warpctc/hip.cmake.patch)
endif()

set(WARPCTC_INCLUDE_DIR
    "${WARPCTC_INSTALL_DIR}/include"
    CACHE PATH "Warp-ctc Directory" FORCE)
# Used in unit test test_WarpCTCLayer
set(WARPCTC_LIB_DIR
    "${WARPCTC_INSTALL_DIR}/lib"
    CACHE PATH "Warp-ctc Library Directory" FORCE)

if(WIN32)
  set(WARPCTC_LIBRARIES
      "${WARPCTC_INSTALL_DIR}/bin/warpctc${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "Warp-ctc Library" FORCE)
else()
  set(WARPCTC_LIBRARIES
      "${WARPCTC_INSTALL_DIR}/lib/libwarpctc${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "Warp-ctc Library" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
   OR WIN32)
  set(USE_OMP OFF)
else()
  set(USE_OMP ON)
endif()

if(WIN32)
  set(WARPCTC_C_FLAGS $<FILTER:${CMAKE_C_FLAGS},EXCLUDE,/Zc:inline>)
  set(WARPCTC_C_FLAGS_DEBUG $<FILTER:${CMAKE_C_FLAGS_DEBUG},EXCLUDE,/Zc:inline>)
  set(WARPCTC_C_FLAGS_RELEASE
      $<FILTER:${CMAKE_C_FLAGS_RELEASE},EXCLUDE,/Zc:inline>)
  set(WARPCTC_CXX_FLAGS $<FILTER:${CMAKE_CXX_FLAGS},EXCLUDE,/Zc:inline>)
  set(WARPCTC_CXX_FLAGS_RELEASE
      $<FILTER:${CMAKE_CXX_FLAGS_RELEASE},EXCLUDE,/Zc:inline>)
  set(WARPCTC_CXX_FLAGS_DEBUG
      $<FILTER:${CMAKE_CXX_FLAGS_DEBUG},EXCLUDE,/Zc:inline>)
else()
  set(WARPCTC_C_FLAGS ${CMAKE_C_FLAGS})
  set(WARPCTC_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  set(WARPCTC_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  set(WARPCTC_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(WARPCTC_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(WARPCTC_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()

ExternalProject_Add(
  extern_warpctc
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${WARPCTC_PREFIX_DIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND
  COMMAND ${WARPCTC_PATCH_COMMAND}
  COMMAND ${WARPCTC_PATCH_CUDA_COMMAND}
  COMMAND ${WARPCTC_PATHCH_ROCM_COMMAND}
  #BUILD_ALWAYS    1
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${WARPCTC_C_FLAGS}
             -DCMAKE_C_FLAGS_DEBUG=${WARPCTC_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${WARPCTC_C_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS=${WARPCTC_CXX_FLAGS}
             -DCMAKE_CXX_FLAGS_RELEASE=${WARPCTC_CXX_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${WARPCTC_CXX_FLAGS_DEBUG}
             -DCMAKE_INSTALL_PREFIX=${WARPCTC_INSTALL_DIR}
             -DWITH_GPU=${WITH_GPU}
             -DWITH_ROCM=${WITH_ROCM}
             -DWITH_OMP=${USE_OMP}
             -DNVCC_FLAGS_EXTRA=${NVCC_FLAGS_EXTRA}
             -DWITH_TORCH=OFF
             -DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON
             -DBUILD_SHARED=ON
             -DBUILD_TESTS=OFF
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
             ${EXTERNAL_OPTIONAL_ARGS}
             ${WARPCTC_CCBIN_OPTION}
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${WARPCTC_INSTALL_DIR}
  BUILD_BYPRODUCTS ${WARPCTC_LIBRARIES})

message(STATUS "warp-ctc library: ${WARPCTC_LIBRARIES}")
get_filename_component(WARPCTC_LIBRARY_PATH ${WARPCTC_LIBRARIES} DIRECTORY)
include_directories(${WARPCTC_INCLUDE_DIR}
)# For warpctc code to include its headers.

add_library(warpctc INTERFACE)
add_dependencies(warpctc extern_warpctc)
