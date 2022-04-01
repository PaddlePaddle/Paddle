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

INCLUDE(ExternalProject)

SET(ARM_BRPC_PROJECT       "extern_arm_brpc")
IF((NOT DEFINED ARM_BRPC_NAME) OR (NOT DEFINED ARM_BRPC_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(PSLIB_BRPC_VER "0.1.0" CACHE STRING "" FORCE)
  SET(ARM_BRPC_NAME "arm_brpc" CACHE STRING "" FORCE)
  SET(ARM_BRPC_URL "https://arm_brpc.bj.bcebos.com/arm_brpc.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "ARM_BRPC_NAME: ${ARM_BRPC_NAME}, ARM_BRPC_URL: ${ARM_BRPC_URL}")
SET(ARM_BRPC_PREFIX_DIR    "${THIRD_PARTY_PATH}/arm_brpc")
SET(ARM_BRPC_DOWNLOAD_DIR  "${ARM_BRPC_PREFIX_DIR}/src/${ARM_BRPC_PROJECT}")
SET(ARM_BRPC_DST_DIR       "output")
SET(ARM_BRPC_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(ARM_BRPC_INSTALL_DIR   ${ARM_BRPC_INSTALL_ROOT}/${ARM_BRPC_DST_DIR})
SET(ARM_BRPC_ROOT          ${ARM_BRPC_INSTALL_DIR})
SET(ARM_BRPC_INC_DIR       ${ARM_BRPC_ROOT}/include/baidu/rpc)
SET(ARM_BRPC_LIB_DIR       ${ARM_BRPC_ROOT}/lib)
SET(ARM_BRPC_LIB           ${ARM_BRPC_LIB_DIR}/libbdrpc.a)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${ARM_BRPC_ROOT}/lib")

INCLUDE_DIRECTORIES(${ARM_BRPC_INC_DIR})

FILE(WRITE ${ARM_BRPC_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(ARM_BRPC)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${ARM_BRPC_NAME}/include ${ARM_BRPC_NAME}/lib \n"
  "        DESTINATION ${ARM_BRPC_DST_DIR})\n")

ExternalProject_Add(
    ${ARM_BRPC_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${ARM_BRPC_PREFIX_DIR}
    DOWNLOAD_DIR          ${ARM_BRPC_DOWNLOAD_DIR}
    #DOWNLOAD_COMMAND      wget --no-check-certificate ${ARM_BRPC_URL} -c -q -O ${ARM_BRPC_NAME}.tar.gz
    DOWNLOAD_COMMAND      cp /home/wangbin/the_one_ps/baidu/base/baidu-rpc/arm_brpc.tar.gz .
                          && tar zxvf ${ARM_BRPC_NAME}.tar.gz && mv output arm_brpc
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${ARM_BRPC_INSTALL_ROOT}
                          -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${ARM_BRPC_INSTALL_ROOT}
                          -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
    BUILD_BYPRODUCTS      ${ARM_BRPC_LIB}
)

#ADD_DEPENDENCIES(extern_arm_brpc protobuf ssl crypto leveldb gflags glog snappy)
ADD_LIBRARY(arm_brpc STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET arm_brpc PROPERTY IMPORTED_LOCATION ${ARM_BRPC_LIB})
ADD_DEPENDENCIES(arm_brpc ${ARM_BRPC_PROJECT})

add_definitions(-DBRPC_WITH_GLOG)

LIST(APPEND external_project_dependencies arm_brpc)
