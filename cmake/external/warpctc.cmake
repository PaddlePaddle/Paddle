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
set(WARPCTC_REPOSITORY ${GIT_URL}/baidu-research/warp-ctc.git)
set(WARPCTC_TAG 37ece0e1bbe8a0019a63ac7e6462c36591c66a5b)

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

if(WITH_ASCEND OR WITH_ASCEND_CL)
  ExternalProject_Add(
    extern_warpctc
    ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
    GIT_REPOSITORY ${WARPCTC_REPOSITORY}
    GIT_TAG ${WARPCTC_TAG}
    PREFIX ${WARPCTC_PREFIX_DIR}
    #UPDATE_COMMAND  ""
    PATCH_COMMAND ""
    BUILD_ALWAYS 1
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
               -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
               -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
               -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
               -DCMAKE_INSTALL_PREFIX=${WARPCTC_INSTALL_DIR}
               -DWITH_GPU=${WITH_GPU}
               -DWITH_ROCM=${WITH_ROCM}
               -DWITH_OMP=${USE_OMP}
               -DWITH_TORCH=OFF
               -DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON
               -DBUILD_SHARED=ON
               -DBUILD_TESTS=OFF
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
               ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_INSTALL_PREFIX:PATH=${WARPCTC_INSTALL_DIR}
    BUILD_BYPRODUCTS ${WARPCTC_LIBRARIES})
else()
  if(WIN32)
    set(WARPCTC_C_FLAGS $<FILTER:${CMAKE_C_FLAGS},EXCLUDE,/Zc:inline>)
    set(WARPCTC_C_FLAGS_DEBUG
        $<FILTER:${CMAKE_C_FLAGS_DEBUG},EXCLUDE,/Zc:inline>)
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
    ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
    GIT_REPOSITORY ${WARPCTC_REPOSITORY}
    GIT_TAG ${WARPCTC_TAG}
    PREFIX ${WARPCTC_PREFIX_DIR}
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
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
               -DWITH_TORCH=OFF
               -DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON
               -DBUILD_SHARED=ON
               -DBUILD_TESTS=OFF
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
               ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_INSTALL_PREFIX:PATH=${WARPCTC_INSTALL_DIR}
    BUILD_BYPRODUCTS ${WARPCTC_LIBRARIES})
endif()

message(STATUS "warp-ctc library: ${WARPCTC_LIBRARIES}")
get_filename_component(WARPCTC_LIBRARY_PATH ${WARPCTC_LIBRARIES} DIRECTORY)
include_directories(${WARPCTC_INCLUDE_DIR}
)# For warpctc code to include its headers.

add_library(warpctc SHARED IMPORTED GLOBAL)
set_property(TARGET warpctc PROPERTY IMPORTED_LOCATION ${WARPCTC_LIBRARIES})
add_dependencies(warpctc extern_warpctc)
