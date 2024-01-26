# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

project(dependency NONE)

include(ExternalProject)
set(XDNN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/xdnn)
set(XDNN_DATA_TYPES_DIR ${XDNN_INSTALL_DIR}/data_types)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${XDNN_INSTALL_DIR}")
set(XDNN_DOWNLOAD_DIR ${PADDLE_SOURCE_DIR}/third_party/xdnn)

set(XDNN_FILE
    "xdnn_v1.4.2.tar.gz"
    CACHE STRING "" FORCE)
set(XDNN_URL
    "https://github.com/intel/xFasterTransformer/releases/download/IntrinsicGemm/${XDNN_FILE}"
    CACHE STRING "" FORCE)
set(XDNN_URL_MD5 ec80903a388a7684d4853fb0f8d2ff81)
set(XDNN_STATIC_LIB ${XDNN_INSTALL_DIR}/libxdnn_static.a)
set(XDNN_LIB ${XDNN_INSTALL_DIR}/libxdnn.so)

set(XDNN_PROJECT "extern_xdnn")
message(STATUS "XDNN_FILE: ${XDNN_FILE}, XDNN_URL: ${XDNN_URL}")
set(XDNN_PREFIX_DIR ${THIRD_PARTY_PATH}/xdnn)

function(download_xdnn)
  message(STATUS "Downloading ${XDNN_URL} to ${XDNN_DOWNLOAD_DIR}/${XDNN_FILE}")
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(
    DOWNLOAD ${XDNN_URL} ${XDNN_DOWNLOAD_DIR}/${XDNN_FILE}
    EXPECTED_MD5 ${XDNN_URL_MD5}
    TIMEOUT 360
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${XDNN_FILE} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${XDNN_FILE} again"
    )
  endif()
endfunction()

# Download and check xdnn.
# if(EXISTS ${XDNN_DOWNLOAD_DIR}/${XDNN_FILE})
#   # file(MD5 ${XDNN_DOWNLOAD_DIR}/${XDNN_FILE} XDNN_MD5)
#   # if(NOT XDNN_MD5 STREQUAL XDNN_URL_MD5)
#   #   # clean build file
#   #   file(REMOVE_RECURSE ${MKLML_PREFIX_DIR})
#   #   file(REMOVE_RECURSE ${MKLML_INSTALL_DIR})
#   #   download_xdnn()
#   # endif()
# else()
download_xdnn()
# endif()

ExternalProject_Add(
  ${XDNN_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${XDNN_DOWNLOAD_DIR}/${XDNN_FILE}
  URL_HASH MD5=${XDNN_URL_MD5}
  DOWNLOAD_DIR ${XDNN_DOWNLOAD_DIR}
  SOURCE_DIR ${XDNN_INSTALL_DIR}
  PREFIX ${XDNN_PREFIX_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${XDNN_STATIC_LIB}
  BUILD_BYPRODUCTS ${XDNN_LIB})

include_directories(${XDNN_DATA_TYPES_DIR})

add_library(xdnn STATIC IMPORTED GLOBAL)
set_property(TARGET xdnn PROPERTY IMPORTED_LOCATION ${XDNN_STATIC_LIB})
add_dependencies(xdnn ${XDNN_PROJECT})
