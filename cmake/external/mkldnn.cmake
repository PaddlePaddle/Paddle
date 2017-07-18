# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.
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

IF(NOT ${WITH_MKLDNN})
  return()
ENDIF(NOT ${WITH_MKLDNN})

INCLUDE(ExternalProject)

SET(MKLDNN_PROJECT "extern_mkldnn")
SET(MKLDNN_SOURCES_DIR ${THIRD_PARTY_PATH}/mkldnn)
SET(MKLDNN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/mkldnn)
SET(MKLDNN_INCLUDE_DIR "${MKLDNN_INSTALL_DIR}/include" CACHE PATH "mkldnn include directory." FORCE)

# The following magic numbers should be updated regularly to keep latest version
SET(MKLDNN_TAG "v0.9")
SET(MKLDNN_MKL_VER "mklml_lnx_2018.0.20170425")

IF(WIN32)
    MESSAGE(WARNING "It is not supported compiling with mkldnn in windows Paddle yet."
      "Force WITH_MKLDNN=OFF")
    SET(WITH_MKLDNN OFF)
    return()
ELSE(WIN32)
    SET(MKLDNN_LIBRARY "${MKLDNN_INSTALL_DIR}/lib/libmkldnn.so" CACHE FILEPATH "mkldnn library." FORCE)
    MESSAGE(STATUS "Set ${MKLDNN_INSTALL_DIR}/lib to runtime path")
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    #SET(CMAKE_MACOSX_RPATH 1) # hold for MacOS
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKLDNN_INSTALL_DIR}/lib")
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${MKLDNN_INCLUDE_DIR})

SET(MKLDNN_CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(MKLDNN_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

ExternalProject_Add(
    ${MKLDNN_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY    "https://github.com/01org/mkl-dnn.git"
    GIT_TAG           "${MKLDNN_TAG}"
    PREFIX            ${MKLDNN_SOURCES_DIR}
    PATCH_COMMAND     cd <SOURCE_DIR>/scripts && ./prepare_mkl.sh
    UPDATE_COMMAND    ""
    CMAKE_ARGS        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    CMAKE_ARGS        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    CMAKE_ARGS        -DCMAKE_CXX_FLAGS=${MKLDNN_CMAKE_CXX_FLAGS}
    CMAKE_ARGS        -DCMAKE_C_FLAGS=${MKLDNN_CMAKE_C_FLAGS}
    CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR}
    CMAKE_ARGS        -DCMAKE_INSTALL_LIBDIR=${MKLDNN_INSTALL_DIR}/lib
    CMAKE_ARGS        -DCMAKE_BUILD_TYPE=Release
    CMAKE_CACHE_ARGS  -DCMAKE_INSTALL_PREFIX:PATH=${MKLDNN_INSTALL_DIR}
                      -DCMAKE_INSTALL_LIBDIR:PATH=${MKLDNN_INSTALL_DIR}/lib
                      -DCMAKE_BUILD_TYPE:STRING=Release
)

SET(MKL_LITE_DIR ${MKLDNN_SOURCES_DIR}/src/${MKLDNN_PROJECT}/external/${MKLDNN_MKL_VER})
SET(MKL_LITE_INC_DIR ${MKL_LITE_DIR}/include)
SET(MKL_LITE_LIB ${MKL_LITE_DIR}/lib/libmklml_intel.so)
SET(MKL_LITE_LIB_IOMP ${MKL_LITE_DIR}/lib/libiomp5.so)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKL_LITE_DIR}/lib")

ADD_LIBRARY(mkldnn STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIBRARY})
ADD_DEPENDENCIES(mkldnn ${MKLDNN_PROJECT})

LIST(APPEND external_project_dependencies mkldnn)
