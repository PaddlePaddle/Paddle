# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

set(BOX_PS_PROJECT "extern_box_ps")
if((NOT DEFINED BOX_PS_VER) OR (NOT DEFINED BOX_PS_URL))
  message(STATUS "use pre defined download url")
  set(BOX_PS_VER
      "0.1.1"
      CACHE STRING "" FORCE)
  set(BOX_PS_NAME
      "box_ps"
      CACHE STRING "" FORCE)
  set(BOX_PS_DOWNLOAD_FILE
      ${BOX_PS_NAME}.tar.gz
      CACHE STRING "" FORCE)
  set(BOX_PS_URL
      "http://box-ps.gz.bcebos.com/${BOX_PS_DOWNLOAD_FILE}"
      CACHE STRING "" FORCE)
endif()
message(STATUS "BOX_PS_NAME: ${BOX_PS_NAME}, BOX_PS_URL: ${BOX_PS_URL}")
set(BOX_PS_SOURCE_DIR "${THIRD_PARTY_PATH}/box_ps")
set(BOX_PS_DOWNLOAD_DIR
    "${PADDLE_SOURCE_DIR}/third_party/box_ps/${CMAKE_SYSTEM_NAME}")
set(BOX_PS_DST_DIR "box_ps")
set(BOX_PS_INSTALL_ROOT "${THIRD_PARTY_PATH}/install")
set(BOX_PS_INSTALL_DIR ${BOX_PS_INSTALL_ROOT}/${BOX_PS_DST_DIR})
set(BOX_PS_ROOT ${BOX_PS_INSTALL_DIR})
set(BOX_PS_INC_DIR ${BOX_PS_ROOT}/include)
set(BOX_PS_LIB_DIR ${BOX_PS_ROOT}/lib)
set(BOX_PS_LIB ${BOX_PS_LIB_DIR}/libbox_ps.so)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${BOX_PS_ROOT}/lib")

include_directories(${BOX_PS_INC_DIR})

file(
  DOWNLOAD ${BOX_PS_URL} ${BOX_PS_DOWNLOAD_DIR}/${BOX_PS_DOWNLOAD_FILE}
  TLS_VERIFY OFF
  STATUS ERR)

file(
  WRITE ${BOX_PS_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(BOX_PS)\n" "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ./include ./lib \n"
  "        DESTINATION ${BOX_PS_DST_DIR})\n")
ExternalProject_Add(
  ${BOX_PS_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${BOX_PS_DOWNLOAD_DIR}/${BOX_PS_DOWNLOAD_FILE}
  PREFIX ${BOX_PS_SOURCE_DIR}
  DOWNLOAD_DIR ${BOX_PS_DOWNLOAD_DIR}
  SOURCE_DIR ${BOX_PS_INSTALL_DIR}
  UPDATE_COMMAND ""
  COMMAND ${CMAKE_COMMAND} -E copy ${BOX_PS_DOWNLOAD_DIR}/CMakeLists.txt
          ${BOX_PS_INSTALL_DIR}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${BOX_PS_INSTALL_ROOT}
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${BOX_PS_INSTALL_ROOT}
                   -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${BOX_PS_LIB})
add_library(box_ps SHARED IMPORTED GLOBAL)
set_property(TARGET box_ps PROPERTY IMPORTED_LOCATION ${BOX_PS_LIB})
add_dependencies(box_ps ${BOX_PS_PROJECT})
