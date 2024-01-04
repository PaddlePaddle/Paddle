# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT WITH_ONNXRUNTIME)
  return()
endif()

if(WITH_ARM)
  message(SEND_ERROR "The current onnxruntime backend doesn't support ARM cpu")
  return()
endif()

include(ExternalProject)

set(PADDLE2ONNX_PROJECT "extern_paddle2onnx")
set(PADDLE2ONNX_VERSION "1.0.0rc2")
set(PADDLE2ONNX_PREFIX_DIR ${THIRD_PARTY_PATH}/paddle2onnx)
set(PADDLE2ONNX_SOURCE_DIR
    ${THIRD_PARTY_PATH}/paddle2onnx/src/${PADDLE2ONNX_PROJECT})
set(PADDLE2ONNX_INSTALL_DIR ${THIRD_PARTY_PATH}/install/paddle2onnx)
set(PADDLE2ONNX_INC_DIR
    "${PADDLE2ONNX_INSTALL_DIR}/include"
    CACHE PATH "paddle2onnx include directory." FORCE)
set(PADDLE2ONNX_LIB_DIR
    "${PADDLE2ONNX_INSTALL_DIR}/lib"
    CACHE PATH "onnxruntime lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${PADDLE2ONNX_LIB_DIR}")
set(PADDLE2ONNX_DOWNLOAD_DIR
    ${PADDLE_SOURCE_DIR}/third_party/paddle2onnx/${CMAKE_SYSTEM_NAME})

# For PADDLE2ONNX code to include internal headers.
include_directories(${PADDLE2ONNX_INC_DIR})
set(PADDLE2ONNX_LIB_NEW_NAME "libpaddle2onnx${CMAKE_SHARED_LIBRARY_SUFFIX}")
if(APPLE)
  set(PADDLE2ONNX_LIB_NAME
      "libpaddle2onnx.${PADDLE2ONNX_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
  set(PADDLE2ONNX_LIB_NAME
      "libpaddle2onnx${CMAKE_SHARED_LIBRARY_SUFFIX}.${PADDLE2ONNX_VERSION}")
endif()

if(WIN32)
  set(PADDLE2ONNX_LIB
      "${PADDLE2ONNX_INSTALL_DIR}/lib/paddle2onnx.dll"
      CACHE FILEPATH "paddle2onnx library." FORCE)
  set(PADDLE2ONNX_COMPILE_LIB
      "${PADDLE2ONNX_INSTALL_DIR}/lib/paddle2onnx.lib"
      CACHE FILEPATH "paddle2onnx compile library." FORCE)
else()
  set(PADDLE2ONNX_SOURCE_LIB
      "${PADDLE2ONNX_SOURCE_DIR}/lib/${PADDLE2ONNX_LIB_NAME}"
      CACHE FILEPATH "PADDLE2ONNX source library." FORCE)
  set(PADDLE2ONNX_LIB
      "${PADDLE2ONNX_LIB_DIR}/${PADDLE2ONNX_LIB_NAME}"
      CACHE FILEPATH "PADDLE2ONNX library." FORCE)
  set(PADDLE2ONNX_COMPILE_LIB
      ${PADDLE2ONNX_LIB}
      CACHE FILEPATH "paddle2onnx compile library." FORCE)
endif()

if(WIN32)
  set(PADDLE2ONNX_URL
      "${GIT_URL}/PaddlePaddle/Paddle2ONNX/releases/download/v${PADDLE2ONNX_VERSION}/paddle2onnx-win-x64-${PADDLE2ONNX_VERSION}.zip"
  )
  set(PADDLE2ONNX_URL_MD5 "122b864cb57338191a7e9ef5f607c4ba")
  set(PADDLE2ONNX_CACHE_EXTENSION "zip")
elseif(APPLE)
  set(PADDLE2ONNX_URL
      "${GIT_URL}/PaddlePaddle/Paddle2ONNX/releases/download/v${PADDLE2ONNX_VERSION}/paddle2onnx-osx-x86_64-${PADDLE2ONNX_VERSION}.tgz"
  )
  set(PADDLE2ONNX_URL_MD5 "32a4381ff8441b69d58ef0fd6fd919eb")
  set(PADDLE2ONNX_CACHE_EXTENSION "tgz")
else()
  set(PADDLE2ONNX_URL
      "${GIT_URL}/PaddlePaddle/Paddle2ONNX/releases/download/v${PADDLE2ONNX_VERSION}/paddle2onnx-linux-x64-${PADDLE2ONNX_VERSION}.tgz"
  )
  set(PADDLE2ONNX_URL_MD5 "3fbb074987ba241327797f76514e937f")
  set(PADDLE2ONNX_CACHE_EXTENSION "tgz")
endif()

set(PADDLE2ONNX_CACHE_FILENAME
    "${PADDLE2ONNX_VERSION}.${PADDLE2ONNX_CACHE_EXTENSION}")

function(download_paddle2onnx)
  message(
    STATUS
      "Downloading ${PADDLE2ONNX_URL} to ${PADDLE2ONNX_DOWNLOAD_DIR}/${PADDLE2ONNX_CACHE_FILENAME}"
  )
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(
    DOWNLOAD ${PADDLE2ONNX_URL}
    ${PADDLE2ONNX_DOWNLOAD_DIR}/${PADDLE2ONNX_CACHE_FILENAME}
    EXPECTED_MD5 ${PADDLE2ONNX_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${PADDLE2ONNX_CACHE_FILENAME} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${PADDLE2ONNX_CACHE_FILENAME} again"
    )
  endif()
endfunction()

if(EXISTS ${PADDLE2ONNX_DOWNLOAD_DIR}/${PADDLE2ONNX_CACHE_FILENAME})
  file(MD5 ${PADDLE2ONNX_DOWNLOAD_DIR}/${PADDLE2ONNX_CACHE_FILENAME}
       PADDLE2ONNX_MD5)
  if(NOT PADDLE2ONNX_MD5 STREQUAL PADDLE2ONNX_URL_MD5)
    download_paddle2onnx()
  endif()
else()
  download_paddle2onnx()
endif()

if(WIN32)
  ExternalProject_Add(
    ${PADDLE2ONNX_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${PADDLE2ONNX_DOWNLOAD_DIR}/${PADDLE2ONNX_CACHE_FILENAME}
    URL_MD5 ${PADDLE2ONNX_URL_MD5}
    PREFIX ${PADDLE2ONNX_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    DOWNLOAD_DIR ${PADDLE2ONNX_DOWNLOAD_DIR}
    SOURCE_DIR ${PADDLE2ONNX_INSTALL_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${PADDLE2ONNX_COMPILE_LIB})
else()
  ExternalProject_Add(
    ${PADDLE2ONNX_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${PADDLE2ONNX_DOWNLOAD_DIR}/${PADDLE2ONNX_CACHE_FILENAME}
    URL_MD5 ${PADDLE2ONNX_URL_MD5}
    PREFIX ${PADDLE2ONNX_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    DOWNLOAD_DIR ${PADDLE2ONNX_DOWNLOAD_DIR}
    SOURCE_DIR ${PADDLE2ONNX_INSTALL_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${PADDLE2ONNX_COMPILE_LIB})
endif()

add_library(paddle2onnx STATIC IMPORTED GLOBAL)
set_property(TARGET paddle2onnx PROPERTY IMPORTED_LOCATION
                                         ${PADDLE2ONNX_COMPILE_LIB})
add_dependencies(paddle2onnx ${PADDLE2ONNX_PROJECT})
