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

set(ONNXRUNTIME_DOWNLOAD_DIR
    ${PADDLE_SOURCE_DIR}/third_party/onnxruntime/${CMAKE_SYSTEM_NAME})

if(WIN32)
  set(ONNXRUNTIME_URL
      "${GIT_URL}/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip"
  )
  set(ONNXRUNTIME_URL_MD5 f21d6bd1feef15935a5f4e1007797593)
  set(ONNXRUNTIME_CACHE_EXTENSION "zip")
elseif(APPLE)
  set(ONNXRUNTIME_URL
      "${GIT_URL}/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz"
  )
  set(ONNXRUNTIME_URL_MD5 6a6f6b7df97587da59976042f475d3f4)
  set(ONNXRUNTIME_CACHE_EXTENSION "tgz")
else()
  set(ONNXRUNTIME_URL
      "${GIT_URL}/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
  )
  set(ONNXRUNTIME_URL_MD5 ce3f2376854b3da4b483d6989666995a)
  set(ONNXRUNTIME_CACHE_EXTENSION "tgz")
endif()

set(ONNXRUNTIME_CACHE_FILENAME
    "${ONNXRUNTIME_VERSION}.${ONNXRUNTIME_CACHE_EXTENSION}")

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

function(download_onnxruntime)
  message(
    STATUS
      "Downloading ${ONNXRUNTIME_URL} to ${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_CACHE_FILENAME}"
  )
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(
    DOWNLOAD ${ONNXRUNTIME_URL}
    ${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_CACHE_FILENAME}
    EXPECTED_MD5 ${ONNXRUNTIME_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${ONNXRUNTIME_CACHE_FILENAME} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${ONNXRUNTIME_CACHE_FILENAME} again"
    )
  endif()
endfunction()

if(EXISTS ${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_CACHE_FILENAME})
  file(MD5 ${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_CACHE_FILENAME}
       ONNXRUNTIME_MD5)
  if(NOT ONNXRUNTIME_MD5 STREQUAL ONNXRUNTIME_URL_MD5)
    # clean build file
    file(REMOVE_RECURSE ${ONNXRUNTIME_PREFIX_DIR})
    file(REMOVE_RECURSE ${ONNXRUNTIME_INSTALL_DIR})
    download_onnxruntime()
  endif()
else()
  download_onnxruntime()
endif()

if(WIN32)
  ExternalProject_Add(
    ${ONNXRUNTIME_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_CACHE_FILENAME}
    URL_MD5 ${ONNXRUNTIME_URL_MD5}
    PREFIX ${ONNXRUNTIME_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    DOWNLOAD_DIR ${ONNXRUNTIME_DOWNLOAD_DIR}
    SOURCE_DIR ${ONNXRUNTIME_INSTALL_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${ONNXRUNTIME_LIB})
else()
  ExternalProject_Add(
    ${ONNXRUNTIME_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_CACHE_FILENAME}
    URL_MD5 ${ONNXRUNTIME_URL_MD5}
    PREFIX ${ONNXRUNTIME_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    DOWNLOAD_DIR ${ONNXRUNTIME_DOWNLOAD_DIR}
    SOURCE_DIR ${ONNXRUNTIME_INSTALL_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${ONNXRUNTIME_LIB})
endif()

add_library(onnxruntime STATIC IMPORTED GLOBAL)
set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_LIB})
add_dependencies(onnxruntime ${ONNXRUNTIME_PROJECT})

if(WIN32)
  string(REGEX REPLACE "/" "\\\\" SOURCE_PATH "${ONNXRUNTIME_SHARED_LIB}")
  set(ONNX_COPY_PATH
      "${CMAKE_BINARY_DIR}/paddle/third_party/onnx_copy_retry.bat")
  file(
    WRITE ${CMAKE_BINARY_DIR}/paddle/third_party/onnx_copy_retry.bat
    ""
    "set build_times=1\n"
    ":retry\n"
    "ECHO copy_onnx run %build_times% time\n"
    "xcopy ${SOURCE_PATH} %1 /Y\n"
    "if %ERRORLEVEL% NEQ 0 (\n"
    "    set /a build_times=%build_times%+1\n"
    "    if %build_times% GEQ 10 (\n"
    "        exit /b 1\n"
    "    ) else (\n"
    "        goto :retry\n"
    "    )\n"
    ")\n"
    "exit /b 0")
endif()

function(copy_onnx TARGET_NAME)
  # If error of Exitcode0xc000007b happened when a .exe running, copy onnxruntime.dll
  # to the .exe folder.
  if(TARGET ${TARGET_NAME})
    string(REGEX REPLACE "/" "\\\\" DESTINATION_PATH
                         "${CMAKE_CURRENT_BINARY_DIR}")
    add_custom_command(
      TARGET ${TARGET_NAME}
      POST_BUILD
      COMMAND ${ONNX_COPY_PATH} ${DESTINATION_PATH} DEPENDS onnxruntime)
  endif()
endfunction()
