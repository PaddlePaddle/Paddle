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

IF(NOT ${WITH_WBAES})
    return() 
ENDIF(NOT ${WITH_WBAES})

IF(APPLE OR WIN32)
    MESSAGE(WARNING "Mac or Windows is not supported with WBAES in Paddle yet.")
    SET(WITH_WBAES OFF CACHE STRING "Disable WBAES packge in MacOS or Windows" FORCE)
    return()
ENDIF()

INCLUDE(ExternalProject)
SET(WBAES_DST_DIR       "wbaes")
SET(WBAES_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(WBAES_INSTALL_DIR   ${WBAES_INSTALL_ROOT}/${WBAES_DST_DIR})
SET(WBAES_ROOT          ${WBAES_INSTALL_DIR})
SET(WBAES_INC_DIR       ${WBAES_ROOT}/include)
SET(WBAES_LIB_DIR       ${WBAES_ROOT}/lib)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${WBAES_ROOT}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(WBAES_TAG   "v1.0.6" CACHE STRING "" FORCE)
SET(WBAES_URL   "http://paddlepaddledeps.bj.bcebos.com/wbaes-sdk.linux-x86_64.${WBAES_TAG}.tgz" CACHE STRING "" FORCE)
SET(WBAES_LIB   ${WBAES_LIB_DIR}/libwbaes.so)
SET(WBAES_SHARED_LIB   ${WBAES_LIB_DIR}/libwbaes.so)

SET(WBAES_PROJECT       "extern_wbaes")
MESSAGE(STATUS "WBAES_URL: ${WBAES_URL}, WBAES_LIB: ${WBAES_LIB}")
SET(WBAES_SOURCE_DIR    "${THIRD_PARTY_PATH}/wbaes") 
SET(WBAES_DOWNLOAD_DIR  "${WBAES_SOURCE_DIR}/src/${WBAES_PROJECT}")

ExternalProject_Add(
    ${WBAES_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                  ${WBAES_SOURCE_DIR}
    URL                     ${WBAES_URL}
    DOWNLOAD_DIR            ${WBAES_DOWNLOAD_DIR}
    DOWNLOAD_NO_PROGRESS    1
    CONFIGURE_COMMAND       ""
    BUILD_COMMAND           ""
    INSTALL_COMMAND         ""
        ${CMAKE_COMMAND} -E copy_directory ${WBAES_DOWNLOAD_DIR}/include ${WBAES_INC_DIR} &&
        ${CMAKE_COMMAND} -E copy_directory ${WBAES_DOWNLOAD_DIR}/lib ${WBAES_LIB_DIR}
)

INCLUDE_DIRECTORIES(${WBAES_INC_DIR})

ADD_LIBRARY(wbaes SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET wbaes PROPERTY IMPORTED_LOCATION ${WBAES_LIB})
SET_PROPERTY(TARGET wbaes PROPERTY IMPORTED_NO_SONAME 1)
ADD_DEPENDENCIES(wbaes ${WBAES_PROJECT})
