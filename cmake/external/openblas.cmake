# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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

SET(CBLAS_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas)
SET(CBLAS_INSTALL_DIR ${PROJECT_BINARY_DIR}/openblas)

ExternalProject_Add(
    openblas
    GIT_REPOSITORY      "https://github.com/xianyi/OpenBLAS.git"
    GIT_TAG             v0.2.19
    PREFIX              ${CBLAS_SOURCES_DIR}
    INSTALL_DIR         ${CBLAS_INSTALL_DIR}
    BUILD_IN_SOURCE     1
    UPDATE_COMMAND      ""
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       cd ${CBLAS_SOURCES_DIR}/src/openblas && make -j4
    INSTALL_COMMAND     cd ${CBLAS_SOURCES_DIR}/src/openblas && make install PREFIX=<INSTALL_DIR>
)

SET(CBLAS_INCLUDE_DIR "${CBLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)
INCLUDE_DIRECTORIES(${CBLAS_INCLUDE_DIR})

IF(WIN32)
    set(CBLAS_LIBRARIES "${CBLAS_INSTALL_DIR}/lib/openblas.lib" CACHE FILEPATH "openblas library." FORCE)
ELSE(WIN32)
    set(CBLAS_LIBRARIES "${CBLAS_INSTALL_DIR}/lib/libopenblas.a" CACHE FILEPATH "openblas library" FORCE)
ENDIF(WIN32)

LIST(APPEND external_project_dependencies openblas)
