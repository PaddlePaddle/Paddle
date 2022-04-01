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

#find_package(OpenSSL REQUIRED)

#message(STATUS "ssl:" ${OPENSSL_SSL_LIBRARY})
#message(STATUS "crypto:" ${OPENSSL_CRYPTO_LIBRARY})

#ADD_LIBRARY(ssl SHARED IMPORTED GLOBAL)
#SET_PROPERTY(TARGET ssl PROPERTY IMPORTED_LOCATION ${OPENSSL_SSL_LIBRARY})

#ADD_LIBRARY(crypto SHARED IMPORTED GLOBAL)
#SET_PROPERTY(TARGET crypto PROPERTY IMPORTED_LOCATION ${OPENSSL_CRYPTO_LIBRARY})

IF((NOT DEFINED ARM_BRPC_NAME) OR (NOT DEFINED ARM_BRPC_URL))
  SET(ARM_BRPC_VER "0.1.0" CACHE STRING "" FORCE)
  SET(ARM_BRPC_NAME "arm_brpc" CACHE STRING "" FORCE)
  SET(ARM_BRPC_URL "https://arm_brpc.bj.bcebos.com/arm_brpc.tar.gz" CACHE STRING "" FORCE)
ENDIF()

MESSAGE(STATUS "ARM_BRPC_NAME: ${ARM_BRPC_NAME}, ARM_BRPC_URL: ${ARM_BRPC_URL}")
SET(ARM_BRPC_PREFIX_DIR    "${THIRD_PARTY_PATH}/arm_brpc")
SET(ARM_BRPC_PROJECT       "extern_arm_brpc")
SET(ARM_BRPC_DOWNLOAD_DIR  "${ARM_BRPC_PREFIX_DIR}/src/${ARM_BRPC_PROJECT}")
SET(ARM_BRPC_DST_DIR       "base/baidu-rpc/output")
SET(ARM_BRPC_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(ARM_BRPC_INSTALL_DIR   ${ARM_BRPC_INSTALL_ROOT}/${ARM_BRPC_DST_DIR})
SET(ARM_BRPC_ROOT          ${ARM_BRPC_INSTALL_DIR})
SET(ARM_BRPC_INC_DIR       ${ARM_BRPC_ROOT}/include/baidu/rpc)
SET(ARM_BRPC_LIB_DIR       ${ARM_BRPC_ROOT}/lib)
SET(ARM_BRPC_LIB           ${ARM_BRPC_LIB_DIR}/libbdrpc.a)

FILE(WRITE ${ARM_BRPC_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(ARM_BRPC)\n"
  "cmake_minimum_required(VERSION 3.0)\n")  # do nothing

execute_process(COMMAND cp /home/wangbin44/Paddle/build/arm_brpc.tar.gz .)
execute_process(COMMAND tar zxvf ${ARM_BRPC_NAME}.tar.gz)
execute_process(COMMAND mkdir install)
execute_process(COMMAND mv base ./install/base)

ADD_LIBRARY(arm_brpc STATIC IMPORTED GLOBAL)  # 直接导入已经生成的库
SET_PROPERTY(TARGET arm_brpc PROPERTY IMPORTED_LOCATION ${ARM_BRPC_LIB})
ADD_DEPENDENCIES(arm_brpc ${ARM_BRPC_PROJECT})

add_definitions(-DBRPC_WITH_GLOG)

LIST(APPEND external_project_dependencies arm_brpc)
