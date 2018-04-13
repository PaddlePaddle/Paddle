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

IF(NOT ${WITH_TENSORRT})
  return()
ENDIF(NOT ${WITH_TENSORRT})

IF(WIN32 OR APPLE)
    MESSAGE(WARNING
        "Windows or Mac is not supported with TENSORRT in Paddle yet."
        "Force WITH_TENSORRT=OFF")
    SET(WITH_TENSORRT OFF CACHE STRING "Disable TENSORRT package in Windows and MacOS" FORCE)
    return()
ENDIF()

INCLUDE(ExternalProject)

SET(TENSORRT_PROJECT       "extern_tensorrt")
SET(TENSORRT_VER           "TensorRT-4.0.0.3.Ubuntu-16.04.4.x86_64-gnu.cuda-8.0.cudnn7.0")
SET(TENSORRT_URL           "http://paddlepaddledeps.bj.bcebos.com/${TENSORRT_VER}.tar.gz")
SET(TENSORRT_SOURCE_DIR    "${THIRD_PARTY_PATH}/tensorrt")
SET(TENSORRT_DOWNLOAD_DIR  "${TENSORRT_SOURCE_DIR}/src/${TENSORRT_PROJECT}")
SET(TENSORRT_DST_DIR       "tensorrt")
SET(TENSORRT_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(TENSORRT_INSTALL_DIR   ${TENSORRT_INSTALL_ROOT}/${TENSORRT_DST_DIR})
SET(TENSORRT_ROOT          ${TENSORRT_INSTALL_DIR})
SET(TENSORRT_INC_DIR       ${TENSORRT_ROOT}/include)

SET(TENSORRT_LIB_DIR       ${TENSORRT_ROOT}/lib)
SET(TENSORRT_LIB           ${TENSORRT_LIB_DIR}/libnvinfer.so)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${TENSORRT_ROOT}/lib")

INCLUDE_DIRECTORIES(${TENSORRT_INC_DIR})

FILE(WRITE ${TENSORRT_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(TENSORRT)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY TensorRT/include TensorRT/lib \n"
  "        DESTINATION ${TENSORRT_DST_DIR})\n")

ExternalProject_Add(
    ${TENSORRT_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${TENSORRT_SOURCE_DIR}
    DOWNLOAD_DIR          ${TENSORRT_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${TENSORRT_URL} -c -q -O ${TENSORRT_VER}.tar.gz
                          && tar xzf ${TENSORRT_VER}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${TENSORRT_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${TENSORRT_INSTALL_ROOT}
)

ADD_LIBRARY(tensorrt SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET tensorrt PROPERTY IMPORTED_LOCATION ${TENSORRT_LIB})
ADD_DEPENDENCIES(tensorrt ${TENSORRT_PROJECT})
LIST(APPEND external_project_dependencies tensorrt)
