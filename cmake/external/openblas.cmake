# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

set(CBLAS_PREFIX_DIR ${THIRD_PARTY_PATH}/openblas)
set(CBLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openblas)
set(CBLAS_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/openblas)
set(CBLAS_TAG v0.3.7)

if(UNIX
   AND NOT APPLE
   AND NOT WITH_ROCM
   AND NOT WITH_XPU)
  set(CBLAS_TAG v0.3.28)
endif()

if(APPLE AND WITH_ARM)
  set(CBLAS_TAG v0.3.13)
endif()

if(WITH_MIPS)
  set(CBLAS_TAG v0.3.13)
endif()

if(WITH_LOONGARCH)
  set(CBLAS_TAG v0.3.18)
endif()

file(GLOB CBLAS_SOURCE_FILE_LIST ${CBLAS_SOURCE_DIR})
list(LENGTH CBLAS_SOURCE_FILE_LIST RES_LEN)
if(RES_LEN EQUAL 0)
  execute_process(COMMAND ${GIT_EXECUTABLE} clone -b ${CBLAS_TAG}
                          "${GIT_URL}/xianyi/OpenBLAS.git" ${CBLAS_SOURCE_DIR})
else()
  # check git tag
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --abbrev=6 --always --tags
    OUTPUT_VARIABLE VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    WORKING_DIRECTORY ${CBLAS_SOURCE_DIR})
  if(NOT ${VERSION} STREQUAL ${CBLAS_TAG})
    message(
      WARNING "openblas version is not ${VERSION}, checkout to ${CBLAS_TAG}")
    execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${CBLAS_TAG}
                    WORKING_DIRECTORY ${CBLAS_SOURCE_DIR})
  endif()
endif()

if(NOT WIN32)
  set(CBLAS_LIBRARIES
      "${CBLAS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
      CACHE FILEPATH "openblas library." FORCE)
  set(CBLAS_INC_DIR
      "${CBLAS_INSTALL_DIR}/include"
      CACHE PATH "openblas include directory." FORCE)
  set(OPENBLAS_CC
      "${CMAKE_C_COMPILER} -Wno-unused-but-set-variable -Wno-unused-variable")

  if(APPLE)
    set(OPENBLAS_CC "${CMAKE_C_COMPILER} -isysroot ${CMAKE_OSX_SYSROOT}")
  endif()
  set(OPTIONAL_ARGS "")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^x86(_64)?$")
    set(OPTIONAL_ARGS DYNAMIC_ARCH=1 NUM_THREADS=64)
  endif()

  if(WITH_ARM)
    set(ARM_ARGS TARGET=ARMV8)
  endif()
  set(COMMON_ARGS CC=${OPENBLAS_CC} NO_SHARED=1 NO_LAPACK=1 libs)
  ExternalProject_Add(
    extern_openblas
    ${EXTERNAL_PROJECT_LOG_ARGS}
    SOURCE_DIR ${CBLAS_SOURCE_DIR}
    PREFIX ${CBLAS_PREFIX_DIR}
    INSTALL_DIR ${CBLAS_INSTALL_DIR}
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make ${ARM_ARGS} -s -j${NPROC} ${COMMON_ARGS} ${OPTIONAL_ARGS}
    INSTALL_COMMAND make install NO_SHARED=1 NO_LAPACK=1 PREFIX=<INSTALL_DIR>
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_BYPRODUCTS ${CBLAS_LIBRARIES})
else()
  set(CBLAS_LIBRARIES
      "${CBLAS_INSTALL_DIR}/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
      CACHE FILEPATH "openblas library." FORCE)
  set(CBLAS_INC_DIR
      "${CBLAS_INSTALL_DIR}/include/openblas"
      CACHE PATH "openblas include directory." FORCE)
  ExternalProject_Add(
    extern_openblas
    ${EXTERNAL_PROJECT_LOG_ARGS}
    SOURCE_DIR ${CBLAS_SOURCE_DIR}
    PREFIX ${CBLAS_PREFIX_DIR}
    INSTALL_DIR ${CBLAS_INSTALL_DIR}
    BUILD_IN_SOURCE 0
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
               -DCMAKE_INSTALL_PREFIX=${CBLAS_INSTALL_DIR}
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
               -DBUILD_SHARED_LIBS=ON
               -DCMAKE_VERBOSE_MAKEFILE=OFF
               -DMSVC_STATIC_CRT=${MSVC_STATIC_CRT}
               ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${CBLAS_INSTALL_DIR}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    # ninja need to know where openblas.lib comes from
    BUILD_BYPRODUCTS ${CBLAS_LIBRARIES})
  set(OPENBLAS_SHARED_LIB
      ${CBLAS_INSTALL_DIR}/bin/openblas${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()
