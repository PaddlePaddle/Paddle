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

IF(NOT ${WITH_MKLML})
  return()
ENDIF(NOT ${WITH_MKLML})

IF(WIN32 OR APPLE)
    MESSAGE(WARNING
        "Windows or Mac is not supported with MKLML in Paddle yet."
        "Force WITH_MKLML=OFF")
    SET(WITH_MKLML OFF CACHE STRING "Disable MKLML package in Windows and MacOS" FORCE)
    return()
ENDIF()

INCLUDE(ExternalProject)

SET(MKLML_PROJECT       "extern_mklml")
IF((NOT DEFINED MKLML_VER) OR (NOT DEFINED MKLML_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(MKLML_VER "mklml_lnx_2018.0.3.20180406" CACHE STRING "" FORCE)
  SET(MKLML_URL "http://paddlepaddledeps.cdn.bcebos.com/${MKLML_VER}.tgz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "MKLML_VER: ${MKLML_VER}, MKLML_URL: ${MKLML_URL}")
SET(MKLML_SOURCE_DIR    "${THIRD_PARTY_PATH}/mklml")
SET(MKLML_DOWNLOAD_DIR  "${MKLML_SOURCE_DIR}/src/${MKLML_PROJECT}")
SET(MKLML_DST_DIR       "mklml")
SET(MKLML_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(MKLML_INSTALL_DIR   ${MKLML_INSTALL_ROOT}/${MKLML_DST_DIR})
SET(MKLML_ROOT          ${MKLML_INSTALL_DIR})
SET(MKLML_INC_DIR       ${MKLML_ROOT}/include)
SET(MKLML_LIB_DIR       ${MKLML_ROOT}/lib)
SET(MKLML_LIB           ${MKLML_LIB_DIR}/libmklml_intel.so)
SET(MKLML_IOMP_LIB      ${MKLML_LIB_DIR}/libiomp5.so)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKLML_ROOT}/lib")

INCLUDE_DIRECTORIES(${MKLML_INC_DIR})

FILE(WRITE ${MKLML_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(MKLML)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${MKLML_VER}/include ${MKLML_VER}/lib \n"
  "        DESTINATION ${MKLML_DST_DIR})\n")

ExternalProject_Add(
    ${MKLML_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${MKLML_SOURCE_DIR}
    DOWNLOAD_DIR          ${MKLML_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${MKLML_URL} -c -q -O ${MKLML_VER}.tgz 
                          && tar zxf ${MKLML_VER}.tgz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${MKLML_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${MKLML_INSTALL_ROOT}
)

ADD_LIBRARY(mklml SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mklml PROPERTY IMPORTED_LOCATION ${MKLML_LIB})
ADD_DEPENDENCIES(mklml ${MKLML_PROJECT})
LIST(APPEND external_project_dependencies mklml)

IF(WITH_C_API)
  INSTALL(FILES ${MKLML_LIB} ${MKLML_IOMP_LIB} DESTINATION lib)
ENDIF()
