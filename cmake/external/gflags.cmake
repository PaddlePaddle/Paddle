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

set(GFLAGS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gflags)
set(GFLAGS_PREFIX_DIR ${THIRD_PARTY_PATH}/gflags)
set(GFLAGS_INCLUDE_DIR
    "${GFLAGS_INSTALL_DIR}/include"
    CACHE PATH "gflags include directory." FORCE)
set(GFLAGS_TAG "v2.2.2")
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/gflags)
if(WIN32)
  set(GFLAGS_LIBRARIES
      "${GFLAGS_INSTALL_DIR}/lib/gflags_static.lib"
      CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
else()
  set(GFLAGS_LIBRARIES
      "${GFLAGS_INSTALL_DIR}/lib/libgflags.a"
      CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
  set(BUILD_COMMAND ${CMAKE_COMMAND} --build .)
  set(INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install)
endif()

include_directories(${GFLAGS_INCLUDE_DIR})

if(WITH_ARM_BRPC)
  set(ARM_GFLAGS_URL
      "https://paddlerec.bj.bcebos.com/online_infer/arm_brpc_ubuntu18/arm_gflags.tar.gz"
      CACHE STRING "" FORCE)
  set(GFLAGS_SOURCE_DIR ${THIRD_PARTY_PATH}/gflags/src/extern_gflags)
  file(
    WRITE ${GFLAGS_SOURCE_DIR}/CMakeLists.txt
    "PROJECT(ARM_GFLAGS)\n" "cmake_minimum_required(VERSION 3.0)\n"
    "install(DIRECTORY arm_gflags/bin  arm_gflags/include arm_gflags/lib \n"
    "        DESTINATION . USE_SOURCE_PERMISSIONS)\n")
  ExternalProject_Add(
    extern_gflags
    ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
    PREFIX ${GFLAGS_PREFIX_DIR}
    DOWNLOAD_DIR ${GFLAGS_SOURCE_DIR}
    DOWNLOAD_COMMAND rm -rf arm_gflags.tar.gz && wget --no-check-certificate
                     ${ARM_GFLAGS_URL} && tar zxvf arm_gflags.tar.gz
    #DOWNLOAD_COMMAND    cp /home/wangbin44/Paddle/build/arm_gflags.tar.gz .
    #                    && tar zxvf arm_gflags.tar.gz
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
               -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    BUILD_BYPRODUCTS ${GFLAGS_LIBRARIES})
else()
  ExternalProject_Add(
    extern_gflags
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX ${GFLAGS_PREFIX_DIR}
    SOURCE_DIR ${SOURCE_DIR}
    UPDATE_COMMAND ""
    BUILD_COMMAND ${BUILD_COMMAND}
    INSTALL_COMMAND ${INSTALL_COMMAND}
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
               -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
               -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
               -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
               -DBUILD_STATIC_LIBS=ON
               -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DBUILD_TESTING=OFF
               -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
               ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    BUILD_BYPRODUCTS ${GFLAGS_LIBRARIES})
endif()

add_library(gflags STATIC IMPORTED GLOBAL)
set_property(TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARIES})
add_dependencies(gflags extern_gflags)

# On Windows (including MinGW), the Shlwapi library is used by gflags if available.
if(WIN32)
  include(CheckIncludeFileCXX)
  check_include_file_cxx("shlwapi.h" HAVE_SHLWAPI)
  if(HAVE_SHLWAPI)
    set_property(GLOBAL PROPERTY OS_DEPENDENCY_MODULES shlwapi.lib)
  endif()
endif()

# We have implemented a custom flags tool paddle_flags to replace gflags.
# User can also choose to use gflags by setting WITH_GFLAGS=ON. But when
# using paddle_flags, gflags is also needed for other third party libraries
# including glog and brpc. So we can not remove gflags completely.
set(flags_dep)
if(WITH_GFLAGS)
  list(APPEND flags_dep gflags)
  add_definitions(-DPADDLE_WITH_GFLAGS)
else()
  list(APPEND flags_dep paddle_flags)
endif()
