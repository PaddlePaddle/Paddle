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

include(ExternalProject)

set(LIBMCT_PROJECT "extern_libmct")
set(LIBMCT_VER
    "0.1.0"
    CACHE STRING "" FORCE)
set(LIBMCT_NAME
    "libmct"
    CACHE STRING "" FORCE)
set(LIBMCT_DOWNLOAD_FILE
    "${LIBMCT_NAME}.tar.gz"
    CACHE STRING "" FORCE)
set(LIBMCT_URL
    "https://pslib.bj.bcebos.com/libmct/${LIBMCT_DOWNLOAD_FILE}"
    CACHE STRING "" FORCE)
set(LIBMCT_URL_MD5 7e6b6c91b45b7490186f7120ef7e08fe)

message(STATUS "LIBMCT_NAME: ${LIBMCT_NAME}, LIBMCT_URL: ${LIBMCT_URL}")
set(LIBMCT_PREFIX_DIR "${THIRD_PARTY_PATH}/libmct")
set(LIBMCT_DOWNLOAD_DIR
    ${PADDLE_SOURCE_DIR}/third_party/libmct/${CMAKE_SYSTEM_NAME})
set(LIBMCT_DST_DIR "libmct")
set(LIBMCT_INSTALL_ROOT "${THIRD_PARTY_PATH}/install")
set(LIBMCT_INSTALL_DIR ${LIBMCT_INSTALL_ROOT}/${LIBMCT_DST_DIR})
set(LIBMCT_ROOT ${LIBMCT_INSTALL_DIR})
set(LIBMCT_INC_DIR ${LIBMCT_ROOT}/include)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${LIBMCT_ROOT}/lib")

include_directories(${LIBMCT_INC_DIR})

function(download_libmct)
  message(
    STATUS
      "Downloading ${LIBMCT_URL} to ${LIBMCT_DOWNLOAD_DIR}/${LIBMCT_DOWNLOAD_FILE}"
  )
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(
    DOWNLOAD ${LIBMCT_URL} ${LIBMCT_DOWNLOAD_DIR}/${LIBMCT_DOWNLOAD_FILE}
    EXPECTED_MD5 ${LIBMCT_URL_MD5}
    TLS_VERIFY OFF
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${LIBMCT_DOWNLOAD_FILE} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${LIBMCT_DOWNLOAD_FILE} again"
    )
  endif()
endfunction()

# Download and check libmct.
if(EXISTS ${LIBMCT_DOWNLOAD_DIR}/${LIBMCT_DOWNLOAD_FILE})
  file(MD5 ${LIBMCT_DOWNLOAD_DIR}/${LIBMCT_DOWNLOAD_FILE} LIBMCT_MD5)
  if(NOT LIBMCT_MD5 STREQUAL LIBMCT_URL_MD5)
    # clean build file
    file(REMOVE_RECURSE ${LIBMCT_PREFIX_DIR})
    file(REMOVE_RECURSE ${LIBMCT_INSTALL_DIR})
    download_libmct()
  endif()
else()
  download_libmct()
endif()

file(
  WRITE ${LIBMCT_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(LIBMCT)\n" "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ./include ./lib \n"
  "        DESTINATION ${LIBMCT_DST_DIR})\n")

ExternalProject_Add(
  ${LIBMCT_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${LIBMCT_DOWNLOAD_DIR}/${LIBMCT_DOWNLOAD_FILE}
  PREFIX ${LIBMCT_PREFIX_DIR}
  DOWNLOAD_DIR ${LIBMCT_DOWNLOAD_DIR}
  SOURCE_DIR ${LIBMCT_INSTALL_DIR}
  UPDATE_COMMAND ""
  COMMAND ${CMAKE_COMMAND} -E copy ${LIBMCT_DOWNLOAD_DIR}/CMakeLists.txt
          ${LIBMCT_INSTALL_DIR}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBMCT_INSTALL_ROOT}
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${LIBMCT_INSTALL_ROOT})

add_library(libmct INTERFACE)

add_dependencies(libmct ${LIBMCT_PROJECT})
