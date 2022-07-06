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

add_definitions(-DPADDLE_WITH_ONNXRUNTIME)

set(ONNXRUNTIME_PROJECT "extern_onnxruntime")
set(ONNXRUNTIME_VERSION "1.11.1")
set(ONNXRUNTIME_PREFIX_DIR ${THIRD_PARTY_PATH}/onnxruntime)
set(ONNXRUNTIME_SOURCE_DIR
    ${THIRD_PARTY_PATH}/onnxruntime/src/${ONNXRUNTIME_PROJECT})
set(ONNXRUNTIME_INSTALL_DIR ${THIRD_PARTY_PATH}/install/onnxruntime)
set(ONNXRUNTIME_INC_DIR
    "${ONNXRUNTIME_INSTALL_DIR}/include"
    CACHE PATH "onnxruntime include directory." FORCE)
set(ONNXRUNTIME_LIB_DIR
    "${ONNXRUNTIME_INSTALL_DIR}/lib"
    CACHE PATH "onnxruntime lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${ONNXRUNTIME_LIB_DIR}")

if(WIN32)
  set(ONNXRUNTIME_URL
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip"
  )
elseif(APPLE)
  set(ONNXRUNTIME_URL
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz"
  )
else()
  set(ONNXRUNTIME_URL
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
  )
endif()

# For ONNXRUNTIME code to include internal headers.
include_directories(${ONNXRUNTIME_INC_DIR})

set(ONNXRUNTIME_LIB_NEW_NAME "libonnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}")
if(APPLE)
  set(ONNXRUNTIME_LIB_NAME
      "libonnxruntime.${ONNXRUNTIME_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
  set(ONNXRUNTIME_LIB_NAME
      "libonnxruntime${CMAKE_SHARED_LIBRARY_SUFFIX}.${ONNXRUNTIME_VERSION}")
endif()
if(WIN32)
  set(ONNXRUNTIME_SOURCE_LIB
      "${ONNXRUNTIME_SOURCE_DIR}/lib/onnxruntime.dll"
      CACHE FILEPATH "ONNXRUNTIME source library." FORCE)
  set(ONNXRUNTIME_SHARED_LIB
      "${ONNXRUNTIME_INSTALL_DIR}/lib/onnxruntime.dll"
      CACHE FILEPATH "ONNXRUNTIME shared library." FORCE)
  set(ONNXRUNTIME_LIB
      "${ONNXRUNTIME_INSTALL_DIR}/lib/onnxruntime.lib"
      CACHE FILEPATH "ONNXRUNTIME static library." FORCE)
elseif(APPLE)
  set(ONNXRUNTIME_SOURCE_LIB
      "${ONNXRUNTIME_SOURCE_DIR}/lib/${ONNXRUNTIME_LIB_NAME}"
      CACHE FILEPATH "ONNXRUNTIME source library." FORCE)
  set(ONNXRUNTIME_LIB
      "${ONNXRUNTIME_INSTALL_DIR}/lib/${ONNXRUNTIME_LIB_NAME}"
      CACHE FILEPATH "ONNXRUNTIME static library." FORCE)
  set(ONNXRUNTIME_SHARED_LIB
      ${ONNXRUNTIME_LIB}
      CACHE FILEPATH "ONNXRUNTIME shared library." FORCE)
else()
  set(ONNXRUNTIME_SOURCE_LIB
      "${ONNXRUNTIME_SOURCE_DIR}/lib/${ONNXRUNTIME_LIB_NAME}"
      CACHE FILEPATH "ONNXRUNTIME source library." FORCE)
  set(ONNXRUNTIME_LIB
      "${ONNXRUNTIME_INSTALL_DIR}/lib/${ONNXRUNTIME_LIB_NAME}"
      CACHE FILEPATH "ONNXRUNTIME static library." FORCE)
  set(ONNXRUNTIME_SHARED_LIB
      ${ONNXRUNTIME_LIB}
      CACHE FILEPATH "ONNXRUNTIME shared library." FORCE)
endif()

if(WIN32)
  ExternalProject_Add(
    ${ONNXRUNTIME_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${ONNXRUNTIME_URL}
    PREFIX ${ONNXRUNTIME_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SOURCE_LIB}
      ${ONNXRUNTIME_SHARED_LIB} && ${CMAKE_COMMAND} -E copy
      ${ONNXRUNTIME_SOURCE_DIR}/lib/onnxruntime.lib ${ONNXRUNTIME_LIB} &&
      ${CMAKE_COMMAND} -E copy_directory ${ONNXRUNTIME_SOURCE_DIR}/include
      ${ONNXRUNTIME_INC_DIR}
    BUILD_BYPRODUCTS ${ONNXRUNTIME_LIB})
else()
  ExternalProject_Add(
    ${ONNXRUNTIME_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${ONNXRUNTIME_URL}
    PREFIX ${ONNXRUNTIME_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SOURCE_LIB} ${ONNXRUNTIME_LIB} &&
      ${CMAKE_COMMAND} -E copy_directory ${ONNXRUNTIME_SOURCE_DIR}/include
      ${ONNXRUNTIME_INC_DIR} && ${CMAKE_COMMAND} -E create_symlink
      ${ONNXRUNTIME_LIB_NAME} ${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NEW_NAME}
    BUILD_BYPRODUCTS ${ONNXRUNTIME_LIB})
endif()

add_library(onnxruntime STATIC IMPORTED GLOBAL)
set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_LIB})
add_dependencies(onnxruntime ${ONNXRUNTIME_PROJECT})

function(copy_onnx TARGET_NAME)
  # If error of Exitcode0xc000007b happened when a .exe running, copy onnxruntime.dll
  # to the .exe folder.
  if(WITH_TESTING AND NOT "$ENV{CI_SKIP_CPP_TEST}" STREQUAL "ON")
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll
      COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SHARED_LIB}
              ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS onnxruntime)
    add_custom_target(copy_onnx_${TARGET_NAME} ALL
                      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll)
    add_dependencies(${TARGET_NAME} copy_onnx_${TARGET_NAME})
  endif()
endfunction()
