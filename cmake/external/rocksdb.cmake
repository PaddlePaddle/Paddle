# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

SET(ROCKSDB_PROJECT       "extern_rocksdb")
IF((NOT DEFINED ROCKSDB_VER) OR (NOT DEFINED ROCKSDB_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(ROCKSDB_VER "master" CACHE STRING "" FORCE)
  SET(ROCKSDB_NAME "rocksdb" CACHE STRING "" FORCE)
  SET(ROCKSDB_URL "https://paddle.bj.bcebos.com/rocksdb.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "ROCKSDB_NAME: ${ROCKSDB_NAME}, ROCKSDB_URL: ${ROCKSDB_URL}")
SET(ROCKSDB_SOURCE_DIR    "${THIRD_PARTY_PATH}/rocksdb")
SET(ROCKSDB_DOWNLOAD_DIR  "${ROCKSDB_SOURCE_DIR}/src/${ROCKSDB_PROJECT}")
SET(ROCKSDB_DST_DIR       "rocksdb")
SET(ROCKSDB_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(ROCKSDB_INSTALL_DIR   ${ROCKSDB_INSTALL_ROOT}/${ROCKSDB_DST_DIR})
SET(ROCKSDB_ROOT          ${ROCKSDB_INSTALL_DIR})
SET(ROCKSDB_INC_DIR       ${ROCKSDB_ROOT}/include)
SET(ROCKSDB_LIB_DIR       ${ROCKSDB_ROOT}/lib)
SET(ROCKSDB_LIB           ${ROCKSDB_LIB_DIR}/librocksdb.a)
#SET(ROCKSDB_IOMP_LIB      ${ROCKSDB_LIB_DIR}/libiomp5.so) #todo what is this
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${ROCKSDB_ROOT}/lib")

INCLUDE_DIRECTORIES(${ROCKSDB_INC_DIR})

FILE(WRITE ${ROCKSDB_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(ROCKSDB)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${ROCKSDB_NAME}/include ${ROCKSDB_NAME}/lib \n"
  "        DESTINATION ${ROCKSDB_DST_DIR})\n")

ExternalProject_Add(
    ${ROCKSDB_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${ROCKSDB_SOURCE_DIR}
    DOWNLOAD_DIR          ${ROCKSDB_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${ROCKSDB_URL} -c -q -O ${ROCKSDB_NAME}.tar.gz
                          && tar zxvf ${ROCKSDB_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${ROCKSDB_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${ROCKSDB_INSTALL_ROOT}
)

ADD_LIBRARY(rocksdb SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET rocksdb PROPERTY IMPORTED_LOCATION ${ROCKSDB_LIB})
ADD_DEPENDENCIES(rocksdb ${ROCKSDB_PROJECT})
