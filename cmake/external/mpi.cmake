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

INCLUDE(ExternalProject)

SET(MPI_PROJECT       "extern_openmpi")
SET(MPI_VER "1.4.5" CACHE STRING "" FORCE)
SET(MPI_NAME "openmpi" CACHE STRING "" FORCE)
SET(MPI_URL "http://10.156.89.14/package/openmpi-1.4.5.tar.gz" CACHE STRING "" FORCE)
MESSAGE(STATUS "MPI_NAME: ${MPI_NAME}, MPI_URL: ${MPI_URL}")
SET(MPI_SOURCE_DIR    "${THIRD_PARTY_PATH}/openmpi")
SET(MPI_DOWNLOAD_DIR  "${MPI_SOURCE_DIR}/src/${MPI_PROJECT}")
SET(MPI_DST_DIR       "openmpi")
SET(MPI_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(MPI_INSTALL_DIR   ${MPI_INSTALL_ROOT}/${MPI_DST_DIR})
SET(MPI_ROOT          ${MPI_INSTALL_DIR})
SET(MPI_INC_DIR       ${MPI_ROOT}/include)
SET(MPI_LIB_DIR       ${MPI_ROOT}/lib)
SET(MPI_LIB           ${MPI_LIB_DIR}/libmpi.so)
SET(MPI_CXX_LIB       ${MPI_LIB_DIR}/libmpi_cxx.so)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MPI_ROOT}/lib")

INCLUDE_DIRECTORIES(${MPI_INC_DIR})

FILE(WRITE ${MPI_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(MPI)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${MPI_NAME}/include ${MPI_NAME}/lib \n"
  "        DESTINATION ${MPI_DST_DIR})\n")

ExternalProject_Add(
    ${MPI_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${MPI_SOURCE_DIR}
    DOWNLOAD_DIR          ${MPI_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${MPI_URL} -c -q -O ${MPI_NAME}.tar.gz
                          && tar zxvf ${MPI_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${MPI_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${MPI_INSTALL_ROOT}
)

ADD_LIBRARY(mpi SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mpi PROPERTY IMPORTED_LOCATION ${MPI_LIB})
ADD_LIBRARY(mpi_cxx SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mpi_cxx PROPERTY IMPORTED_LOCATION ${MPI_CXX_LIB})
ADD_DEPENDENCIES(mpi mpi_cxx ${MPI_PROJECT})

