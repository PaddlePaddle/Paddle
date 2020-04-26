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

SET(BOX_PS_PROJECT       "extern_box_ps")
IF((NOT DEFINED BOX_PS_VER) OR (NOT DEFINED BOX_PS_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(BOX_PS_VER "0.1.1" CACHE STRING "" FORCE)
  SET(BOX_PS_NAME "box_ps" CACHE STRING "" FORCE)
  SET(BOX_PS_URL "http://box-ps.gz.bcebos.com/box_ps.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "BOX_PS_NAME: ${BOX_PS_NAME}, BOX_PS_URL: ${BOX_PS_URL}")
SET(BOX_PS_SOURCE_DIR    "${THIRD_PARTY_PATH}/box_ps")
SET(BOX_PS_DOWNLOAD_DIR  "${BOX_PS_SOURCE_DIR}/src/${BOX_PS_PROJECT}")
SET(BOX_PS_DST_DIR       "box_ps")
SET(BOX_PS_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(BOX_PS_INSTALL_DIR   ${BOX_PS_INSTALL_ROOT}/${BOX_PS_DST_DIR})
SET(BOX_PS_ROOT          ${BOX_PS_INSTALL_DIR})
SET(BOX_PS_INC_DIR       ${BOX_PS_ROOT}/include)
SET(BOX_PS_LIB_DIR       ${BOX_PS_ROOT}/lib)
SET(BOX_PS_LIB           ${BOX_PS_LIB_DIR}/libbox_ps.so)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${BOX_PS_ROOT}/lib")

INCLUDE_DIRECTORIES(${BOX_PS_INC_DIR})
FILE(WRITE ${BOX_PS_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(BOX_PS)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${BOX_PS_NAME}/include ${BOX_PS_NAME}/lib \n"
  "        DESTINATION ${BOX_PS_DST_DIR})\n")
ExternalProject_Add(
    ${BOX_PS_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${BOX_PS_SOURCE_DIR}
    DOWNLOAD_DIR          ${BOX_PS_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${BOX_PS_URL} -c -q -O ${BOX_PS_NAME}.tar.gz
                          && tar zxvf ${BOX_PS_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${BOX_PS_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${BOX_PS_INSTALL_ROOT}
)
ADD_LIBRARY(box_ps SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET box_ps PROPERTY IMPORTED_LOCATION ${BOX_PS_LIB})
ADD_DEPENDENCIES(box_ps ${BOX_PS_PROJECT})
