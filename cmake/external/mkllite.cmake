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

IF(NOT ${WITH_MKL_LITE})
  return()
ENDIF(NOT ${WITH_MKL_LITE})

INCLUDE(ExternalProject)

SET(MKL_LITE_PROJECT       "extern_mkllite")
SET(MKL_LITE_VER           "mklml_lnx_2018.0.20170425")
SET(MKL_LITE_URL           "https://github.com/01org/mkl-dnn/releases/download/v0.9/${MKL_LITE_VER}.tgz")
SET(MKL_LITE_DOWNLOAD_DIR  ${THIRD_PARTY_PATH}/mkllite)

SET(MKL_LITE_ROOT          ${MKL_LITE_DOWNLOAD_DIR}/${MKL_LITE_VER})
SET(MKL_LITE_INC_DIR       ${MKL_LITE_ROOT}/include)
SET(MKL_LITE_LIB_DIR       ${MKL_LITE_ROOT}/lib)
SET(MKL_LITE_LIB           ${MKL_LITE_LIB_DIR}/libmklml_intel.so)
SET(MKL_LITE_IOMP_LIB      ${MKL_LITE_LIB_DIR}/libiomp5.so)
SET(CMAKE_INSTALL_RPATH    "${CMAKE_INSTALL_RPATH}" "${MKL_LITE_ROOT}/lib")

INCLUDE_DIRECTORIES(${MKL_LITE_INC_DIR})

ExternalProject_Add(
    ${MKL_LITE_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${MKL_LITE_DOWNLOAD_DIR}
    DOWNLOAD_DIR          ${MKL_LITE_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${MKL_LITE_URL}
                          && tar -xzf ${MKL_LITE_DOWNLOAD_DIR}/${MKL_LITE_VER}.tgz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    PATCH_COMMAND         ""
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    INSTALL_COMMAND       ""
    TEST_COMMAND          ""
)

IF (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    SET(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/mkllite_dummy.c)
    FILE(WRITE ${dummyfile} "const char * dummy_mkllite = \"${dummyfile}\";")
    ADD_LIBRARY(mkllite STATIC ${dummyfile})
ELSE()
    ADD_LIBRARY(mkllite INTERFACE)
ENDIF()

ADD_DEPENDENCIES(mkllite ${MKL_LITE_PROJECT})

LIST(APPEND external_project_dependencies mkllite)
