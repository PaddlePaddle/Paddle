# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
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

SET(LIBMCT_PROJECT       "extern_libmct")
IF((NOT DEFINED LIBMCT_VER) OR (NOT DEFINED LIBMCT_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(LIBMCT_VER "0.1.0" CACHE STRING "" FORCE)
  SET(LIBMCT_NAME "libmct" CACHE STRING "" FORCE)
  SET(LIBMCT_URL "https://pslib.bj.bcebos.com/libmct/libmct.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "LIBMCT_NAME: ${LIBMCT_NAME}, LIBMCT_URL: ${LIBMCT_URL}")
SET(LIBMCT_PREFIX_DIR    "${THIRD_PARTY_PATH}/libmct")
SET(LIBMCT_DOWNLOAD_DIR  "${LIBMCT_PREFIX_DIR}/src/${LIBMCT_PROJECT}")
SET(LIBMCT_DST_DIR       "libmct")
SET(LIBMCT_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(LIBMCT_INSTALL_DIR   ${LIBMCT_INSTALL_ROOT}/${LIBMCT_DST_DIR})
SET(LIBMCT_ROOT          ${LIBMCT_INSTALL_DIR})
SET(LIBMCT_INC_DIR       ${LIBMCT_ROOT}/include)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${LIBMCT_ROOT}/lib")

INCLUDE_DIRECTORIES(${LIBMCT_INC_DIR})

FILE(WRITE ${LIBMCT_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(LIBMCT)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${LIBMCT_NAME}/include ${LIBMCT_NAME}/lib \n"
  "        DESTINATION ${LIBMCT_DST_DIR})\n")

ExternalProject_Add(
    ${LIBMCT_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${LIBMCT_PREFIX_DIR}
    DOWNLOAD_DIR          ${LIBMCT_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${LIBMCT_URL} -c -q -O ${LIBMCT_NAME}.tar.gz
                          && tar zxvf ${LIBMCT_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${LIBMCT_INSTALL_ROOT}
                          -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${LIBMCT_INSTALL_ROOT}
                          -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
)

add_library(libmct INTERFACE)

ADD_DEPENDENCIES(libmct ${LIBMCT_PROJECT})
