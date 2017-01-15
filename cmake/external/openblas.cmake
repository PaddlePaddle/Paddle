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

INCLUDE(cblas)

IF(NOT ${CBLAS_FOUND})
    INCLUDE(ExternalProject)

    SET(CBLAS_SOURCES_DIR ${THIRD_PARTY_PATH}/openblas)
    SET(CBLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openblas)
    SET(CBLAS_INC_DIR "${CBLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)

    IF(WIN32)
        SET(CBLAS_LIBRARIES "${CBLAS_INSTALL_DIR}/lib/openblas.lib" CACHE FILEPATH "openblas library." FORCE)
    ELSE(WIN32)
        SET(CBLAS_LIBRARIES "${CBLAS_INSTALL_DIR}/lib/libopenblas.a" CACHE FILEPATH "openblas library" FORCE)
    ENDIF(WIN32)

    IF(CMAKE_COMPILER_IS_GNUCC)
        ENABLE_LANGUAGE(Fortran)
        LIST(APPEND CBLAS_LIBRARIES gfortran pthread)
    ENDIF(CMAKE_COMPILER_IS_GNUCC)

    ExternalProject_Add(
        openblas
        GIT_REPOSITORY      "git://github.com/xianyi/OpenBLAS.git"
        GIT_TAG             "v0.2.19"
        PREFIX              ${CBLAS_SOURCES_DIR}
        INSTALL_DIR         ${CBLAS_INSTALL_DIR}
        BUILD_IN_SOURCE     1
        CONFIGURE_COMMAND   ""
        BUILD_COMMAND       make CC=${CMAKE_C_COMPILER} FC=${CMAKE_Fortran_COMPILER} NO_SHARED=1 libs netlib
        INSTALL_COMMAND     make install NO_SHARED=1 PREFIX=<INSTALL_DIR>
        UPDATE_COMMAND      ""
    )

    LIST(APPEND external_project_dependencies openblas)
ENDIF()

INCLUDE_DIRECTORIES(${CBLAS_INC_DIR})
