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

SET(GLOO_PROJECT       "extern_gloo")
IF((NOT DEFINED GLOO_VER) OR (NOT DEFINED GLOO_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(GLOO_VER "master" CACHE STRING "" FORCE)
  SET(GLOO_NAME "gloo" CACHE STRING "" FORCE)
  SET(GLOO_URL "https://pslib.bj.bcebos.com/gloo.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "GLOO_NAME: ${GLOO_NAME}, GLOO_URL: ${GLOO_URL}")
SET(GLOO_SOURCE_DIR    "${THIRD_PARTY_PATH}/gloo")
SET(GLOO_DOWNLOAD_DIR  "${GLOO_SOURCE_DIR}/src/${GLOO_PROJECT}")
SET(GLOO_DST_DIR       "gloo")
SET(GLOO_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(GLOO_INSTALL_DIR   ${GLOO_INSTALL_ROOT}/${GLOO_DST_DIR})
SET(GLOO_ROOT          ${GLOO_INSTALL_DIR})
SET(GLOO_INC_DIR       ${GLOO_ROOT}/include)
SET(GLOO_LIB_DIR       ${GLOO_ROOT}/lib)
SET(GLOO_LIB           ${GLOO_LIB_DIR}/libgloo.a)
#SET(GLOO_IOMP_LIB      ${GLOO_LIB_DIR}/libiomp5.so) #todo what is this
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${GLOO_ROOT}/lib")

INCLUDE_DIRECTORIES(${GLOO_INC_DIR})

FILE(WRITE ${GLOO_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(GLOO)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${GLOO_NAME}/include ${GLOO_NAME}/lib \n"
  "        DESTINATION ${GLOO_DST_DIR})\n")

ExternalProject_Add(
    ${GLOO_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${GLOO_SOURCE_DIR}
    DOWNLOAD_DIR          ${GLOO_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${GLOO_URL} -c -q -O ${GLOO_NAME}.tar.gz
                          && tar zxvf ${GLOO_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${GLOO_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${GLOO_INSTALL_ROOT}
)

ADD_LIBRARY(gloo SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET gloo PROPERTY IMPORTED_LOCATION ${GLOO_LIB})
ADD_DEPENDENCIES(gloo ${GLOO_PROJECT})
