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

SET(MKLDNN_PROJECT        "extern_mkldnn")
SET(MKLDNN_SOURCES_DIR    ${THIRD_PARTY_PATH}/mkldnn)
SET(MKLDNN_INSTALL_ROOT   ${CMAKE_INSTALL_PREFIX})
IF(NOT "$ENV{HOME}" STREQUAL "/root")
    SET(MKLDNN_INSTALL_ROOT  "$ENV{HOME}")
ENDIF()

SET(MKLDNN_INSTALL_DIR  "${MKLDNN_INSTALL_ROOT}/opt/paddle/third_party/mkldnn")
SET(MKLDNN_INC_DIR      "${MKLDNN_INSTALL_DIR}/include" CACHE PATH "mkldnn include directory." FORCE)

IF(WIN32)
    MESSAGE(WARNING "Windowns is not supported with MKLDNN in Paddle yet."
      "Force WITH_MKLDNN=OFF")
    SET(WITH_MKLDNN OFF CACHE STRING "Disable MKLDNN in Windows" FORCE)
    return()
ELSE(WIN32)
    IF(APPLE)
        MESSAGE(WARNING "MacOS is not supported with MKLDNN in Paddle yet."
            "Force WITH_MKLDNN=OFF")
        SET(WITH_MKLDNN OFF CACHE STRING "Disable MKLDNN in MacOS" FORCE)
        #SET(CMAKE_MACOSX_RPATH 1) # hold for MacOS
        return()
    ENDIF(APPLE)
    SET(MKLDNN_LIB "${MKLDNN_INSTALL_DIR}/lib/libmkldnn.so" CACHE FILEPATH "mkldnn library." FORCE)
    MESSAGE(STATUS "Set ${MKLDNN_INSTALL_DIR}/lib to runtime path")
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKLDNN_INSTALL_DIR}/lib")
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${MKLDNN_INC_DIR})

IF(${CBLAS_PROVIDER} STREQUAL "MKLML")
    SET(MKLDNN_DEPENDS   ${MKLML_PROJECT})
    SET(MKLDNN_MKLROOT   ${MKLML_ROOT})
    SET(MKLDNN_IOMP_LIB  ${MKLML_IOMP_LIB})
    SET(MKLDNN_IOMP_DIR  ${MKLML_LIB_DIR})
ENDIF()

ExternalProject_Add(
    ${MKLDNN_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ${MKLDNN_DEPENDS}
    GIT_REPOSITORY      "https://github.com/01org/mkl-dnn.git"
    GIT_TAG             "v0.9"
    PREFIX              ${MKLDNN_SOURCES_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR}
    CMAKE_ARGS          -DMKLROOT=${MKLDNN_MKLROOT}
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${MKLDNN_INSTALL_DIR}
)

ADD_LIBRARY(mkldnn SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB})
ADD_DEPENDENCIES(mkldnn ${MKLDNN_PROJECT})
MESSAGE(STATUS "Mkldnn library: ${MKLDNN_LIB}")
LIST(APPEND external_project_dependencies mkldnn)
