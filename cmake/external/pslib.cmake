# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

set(PSLIB_PROJECT "extern_pslib")
if((NOT DEFINED PSLIB_VER) OR (NOT DEFINED PSLIB_URL))
  message(STATUS "use pre defined download url")
  set(PSLIB_VER
      "0.1.1"
      CACHE STRING "" FORCE)
  set(PSLIB_NAME
      "pslib"
      CACHE STRING "" FORCE)
  if(NOT WITH_HETERPS)
    set(PSLIB_URL
        "https://pslib.bj.bcebos.com/pslib.tar.gz"
        CACHE STRING "" FORCE)
  else()
    set(PSLIB_URL
        "https://pslib.bj.bcebos.com/pslib_3631b2/pslib.tar.gz"
        CACHE STRING "" FORCE)
  endif()
endif()
message(STATUS "PSLIB_NAME: ${PSLIB_NAME}, PSLIB_URL: ${PSLIB_URL}")
set(PSLIB_PREFIX_DIR "${THIRD_PARTY_PATH}/pslib")
set(PSLIB_DOWNLOAD_DIR "${PSLIB_PREFIX_DIR}/src/${PSLIB_PROJECT}")
set(PSLIB_DST_DIR "pslib")
set(PSLIB_INSTALL_ROOT "${THIRD_PARTY_PATH}/install")
set(PSLIB_INSTALL_DIR ${PSLIB_INSTALL_ROOT}/${PSLIB_DST_DIR})
set(PSLIB_ROOT ${PSLIB_INSTALL_DIR})
set(PSLIB_INC_DIR ${PSLIB_ROOT}/include)
set(PSLIB_LIB_DIR ${PSLIB_ROOT}/lib)
set(PSLIB_LIB ${PSLIB_LIB_DIR}/libps.so)
set(JVM_LIB ${PSLIB_LIB_DIR}/libjvm.so)
set(PSLIB_VERSION_PY ${PSLIB_DOWNLOAD_DIR}/pslib/version.py)
set(PSLIB_IOMP_LIB ${PSLIB_LIB_DIR}/libiomp5.so) #todo what is this
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${PSLIB_ROOT}/lib")

include_directories(${PSLIB_INC_DIR})

file(
  WRITE ${PSLIB_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(PSLIB)\n" "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${PSLIB_NAME}/include ${PSLIB_NAME}/lib \n"
  "        DESTINATION ${PSLIB_DST_DIR})\n")

ExternalProject_Add(
  ${PSLIB_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${PSLIB_PREFIX_DIR}
  DOWNLOAD_DIR ${PSLIB_DOWNLOAD_DIR}
  DOWNLOAD_COMMAND wget --no-check-certificate ${PSLIB_URL} -c -q -O
                   ${PSLIB_NAME}.tar.gz && tar zxvf ${PSLIB_NAME}.tar.gz
  DOWNLOAD_NO_PROGRESS 1
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PSLIB_INSTALL_ROOT}
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${PSLIB_INSTALL_ROOT}
                   -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${PSLIB_LIB} ${JVM_LIB})

add_library(pslib SHARED IMPORTED GLOBAL)
set_property(TARGET pslib PROPERTY IMPORTED_LOCATION ${PSLIB_LIB})
add_dependencies(pslib ${PSLIB_PROJECT})
target_link_libraries(pslib INTERFACE ${PSLIB_LIB_DIR}/libjvm.so)
