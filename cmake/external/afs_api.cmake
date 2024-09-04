# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

set(AFSAPI_PROJECT "extern_afs_api")
if((NOT DEFINED AFSAPI_VER) OR (NOT DEFINED AFSAPI_URL))
  message(STATUS "use pre defined download url")
  set(AFSAPI_VER
      "0.1.1"
      CACHE STRING "" FORCE)
  set(AFSAPI_NAME
      "afs_api"
      CACHE STRING "" FORCE)
  set(AFSAPI_URL
      "https://fleet.bj.bcebos.com/heterps/afs_api.tar.gz"
      CACHE STRING "" FORCE)
endif()
message(STATUS "AFSAPI_NAME: ${AFSAPI_NAME}, AFSAPI_URL: ${AFSAPI_URL}")
set(AFSAPI_PREFIX_DIR "${THIRD_PARTY_PATH}/afs_api")
set(AFSAPI_DOWNLOAD_DIR "${AFSAPI_PREFIX_DIR}/src/${AFSAPI_PROJECT}")
set(AFSAPI_DST_DIR "afs_api")
set(AFSAPI_INSTALL_ROOT "${THIRD_PARTY_PATH}/install")
set(AFSAPI_INSTALL_DIR ${AFSAPI_INSTALL_ROOT}/${AFSAPI_DST_DIR})
set(AFSAPI_ROOT ${AFSAPI_INSTALL_DIR})
set(AFSAPI_INC_DIR ${AFSAPI_ROOT}/include)
set(AFSAPI_LIB_DIR ${AFSAPI_ROOT}/lib)
set(AFSAPI_LIB ${AFSAPI_LIB_DIR}/libafs-api-so.so)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${AFSAPI_ROOT}/lib")

include_directories(${AFSAPI_INC_DIR})

file(
  WRITE ${AFSAPI_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(AFSAPI)\n" "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${AFSAPI_NAME}/include ${AFSAPI_NAME}/lib \n"
  "        DESTINATION ${AFSAPI_DST_DIR})\n")

ExternalProject_Add(
  ${AFSAPI_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${AFSAPI_PREFIX_DIR}
  DOWNLOAD_DIR ${AFSAPI_DOWNLOAD_DIR}
  DOWNLOAD_COMMAND wget --no-check-certificate ${AFSAPI_URL} -c -q -O
                   ${AFSAPI_NAME}.tar.gz && tar zxvf ${AFSAPI_NAME}.tar.gz
  DOWNLOAD_NO_PROGRESS 1
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${AFSAPI_INSTALL_ROOT}
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${AFSAPI_INSTALL_ROOT}
                   -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${AFSAPI_LIB})

add_library(afs_api SHARED IMPORTED GLOBAL)
set_property(TARGET afs_api PROPERTY IMPORTED_LOCATION ${AFSAPI_LIB})
add_dependencies(afs_api ${AFSAPI_PROJECT})
