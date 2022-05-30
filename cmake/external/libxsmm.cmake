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

INCLUDE (ExternalProject)

SET(LIBXSMM_PREFIX_DIR ${THIRD_PARTY_PATH}/libxsmm)
SET(LIBXSMM_INSTALL_DIR ${THIRD_PARTY_PATH}/install/libxsmm)
SET(LIBXSMM_INCLUDE_DIR "${LIBXSMM_INSTALL_DIR}/include" CACHE PATH "LIBXSMM include directory." FORCE)
SET(LIBXSMM_LIBRARY_DIR "${LIBXSMM_INSTALL_DIR}/lib" CACHE PATH "LIBXSMM library directory." FORCE)
SET(LIBXSMM_LIB        "${LIBXSMM_LIBRARY_DIR}/libxsmm.a")
SET(LIBXSMMNOBLAS_LIB  "${LIBXSMM_LIBRARY_DIR}/libxsmmnoblas.a")

ExternalProject_Add(
    extern_libxsmm
    ${SHALLOW_CLONE}
    GIT_REPOSITORY  "${GIT_URL}/hfp/libxsmm.git"
    GIT_TAG         "7cc03b5b342fdbc6b6d990b190671c5dbb8489a2"
    PREFIX          ${LIBXSMM_PREFIX_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND   $(MAKE) --silent PREFIX=${LIBXSMM_INSTALL_DIR} CXX=g++ CC=gcc WARP=0 install
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${LIBXSMM_LIB}
    BUILD_BYPRODUCTS ${LIBXSMMNOBLAS_LIB}
)
ADD_LIBRARY(libxsmm STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET libxsmm PROPERTY IMPORTED_LOCATION "${LIBXSMM_LIB}")
SET_PROPERTY(TARGET libxsmm PROPERTY IMPORTED_LOCATION "${LIBXSMMNOBLAS_LIB}")

MESSAGE(STATUS "Libxsmm library: ${LIBXSMM_LIBS}")
include_directories(${LIBXSMM_INCLUDE_DIR})
ADD_DEFINITIONS(-DPADDLE_WITH_LIBXSMM)
ADD_DEPENDENCIES(libxsmm extern_libxsmm)
