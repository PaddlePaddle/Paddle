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

IF(APPLE)
    MESSAGE(WARNING "Mac is not supported with MKLML in Paddle yet. Force WITH_MKLML=OFF.")
    SET(WITH_MKLML OFF CACHE STRING "Disable MKLML package in MacOS" FORCE)
    return()
ENDIF()

INCLUDE(ExternalProject)
SET(MKLML_DST_DIR       "mklml")
SET(MKLML_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(MKLML_INSTALL_DIR   ${MKLML_INSTALL_ROOT}/${MKLML_DST_DIR})
SET(MKLML_ROOT          ${MKLML_INSTALL_DIR})
SET(MKLML_INC_DIR       ${MKLML_ROOT}/include)
SET(MKLML_LIB_DIR       ${MKLML_ROOT}/lib)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKLML_ROOT}/lib")

SET(TIME_VERSION "2019.0.1.20181227")
IF(WIN32)
    SET(MKLML_VER "mklml_win_${TIME_VERSION}" CACHE STRING "" FORCE)
    SET(MKLML_URL "https://paddlepaddledeps.bj.bcebos.com/${MKLML_VER}.zip" CACHE STRING "" FORCE)
    SET(MKLML_LIB                 ${MKLML_LIB_DIR}/mklml.lib)
    SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5md.lib)
    SET(MKLML_SHARED_LIB          ${MKLML_LIB_DIR}/mklml.dll)
    SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5md.dll)
ELSE()  
    SET(MKLML_VER "mklml_lnx_${TIME_VERSION}" CACHE STRING "" FORCE)
    SET(MKLML_URL "http://paddlepaddledeps.bj.bcebos.com/${MKLML_VER}.tgz" CACHE STRING "" FORCE)
    SET(MKLML_LIB                 ${MKLML_LIB_DIR}/libmklml_intel.so)
    SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5.so)
    SET(MKLML_SHARED_LIB          ${MKLML_LIB_DIR}/libmklml_intel.so)
    SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5.so)
ENDIF()

SET(MKLML_PROJECT       "extern_mklml")
MESSAGE(STATUS "MKLML_VER: ${MKLML_VER}, MKLML_URL: ${MKLML_URL}")
SET(MKLML_SOURCE_DIR    "${THIRD_PARTY_PATH}/mklml")
SET(MKLML_DOWNLOAD_DIR  "${MKLML_SOURCE_DIR}/src/${MKLML_PROJECT}")

ExternalProject_Add(
    ${MKLML_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                 ${MKLML_SOURCE_DIR}
    URL                    ${MKLML_URL}
    DOWNLOAD_DIR          ${MKLML_DOWNLOAD_DIR}
    DOWNLOAD_NO_PROGRESS  1
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_directory ${MKLML_DOWNLOAD_DIR}/include ${MKLML_INC_DIR} &&
        ${CMAKE_COMMAND} -E copy_directory ${MKLML_DOWNLOAD_DIR}/lib ${MKLML_LIB_DIR}
)

INCLUDE_DIRECTORIES(${MKLML_INC_DIR})

ADD_LIBRARY(mklml SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mklml PROPERTY IMPORTED_LOCATION ${MKLML_LIB})
ADD_DEPENDENCIES(mklml ${MKLML_PROJECT})
LIST(APPEND external_project_dependencies mklml)
