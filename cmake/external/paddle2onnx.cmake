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
      "https://github.com/PaddlePaddle/Paddle2ONNX/releases/download/v${PADDLE2ONNX_VERSION}/paddle2onnx-win-x64-${PADDLE2ONNX_VERSION}.zip"
  )
elseif(APPLE)
  set(PADDLE2ONNX_URL
      "https://github.com/PaddlePaddle/Paddle2ONNX/releases/download/v${PADDLE2ONNX_VERSION}/paddle2onnx-osx-x86_64-${PADDLE2ONNX_VERSION}.tgz"
  )
else()
  set(PADDLE2ONNX_URL
      "https://github.com/PaddlePaddle/Paddle2ONNX/releases/download/v${PADDLE2ONNX_VERSION}/paddle2onnx-linux-x64-${PADDLE2ONNX_VERSION}.tgz"
  )
endif()

if(WIN32)
  ExternalProject_Add(
    ${PADDLE2ONNX_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${PADDLE2ONNX_URL}
    PREFIX ${PADDLE2ONNX_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${PADDLE2ONNX_SOURCE_DIR}/lib
      ${PADDLE2ONNX_LIB_DIR} && ${CMAKE_COMMAND} -E copy_directory
      ${PADDLE2ONNX_SOURCE_DIR}/include ${PADDLE2ONNX_INC_DIR}
    BUILD_BYPRODUCTS ${PADDLE2ONNX_COMPILE_LIB})
else()
  ExternalProject_Add(
    ${PADDLE2ONNX_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${PADDLE2ONNX_URL}
    PREFIX ${PADDLE2ONNX_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy ${PADDLE2ONNX_SOURCE_LIB}
      ${PADDLE2ONNX_COMPILE_LIB} && ${CMAKE_COMMAND} -E copy_directory
      ${PADDLE2ONNX_SOURCE_DIR}/include ${PADDLE2ONNX_INC_DIR} &&
      ${CMAKE_COMMAND} -E create_symlink ${PADDLE2ONNX_LIB_NAME}
      ${PADDLE2ONNX_LIB_DIR}/${PADDLE2ONNX_LIB_NEW_NAME}
    BUILD_BYPRODUCTS ${PADDLE2ONNX_COMPILE_LIB})
endif()

add_library(paddle2onnx STATIC IMPORTED GLOBAL)
set_property(TARGET paddle2onnx PROPERTY IMPORTED_LOCATION
                                         ${PADDLE2ONNX_COMPILE_LIB})
add_dependencies(paddle2onnx ${PADDLE2ONNX_PROJECT})
