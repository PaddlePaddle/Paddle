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

    SET(CBLAS_LIBRARIES "${CBLAS_INSTALL_DIR}/lib/${LIBRARY_PREFIX}openblas${STATIC_LIBRARY_SUFFIX}"
        CACHE FILEPATH "openblas library." FORCE)

    SET(COMMON_ARGS CC=${CMAKE_C_COMPILER} NO_SHARED=1 NO_LAPACK=1 libs)

    IF(CMAKE_CROSSCOMPILING)
        IF(ANDROID)
            # arm_soft_fp_abi branch of OpenBLAS to support softfp
            #   https://github.com/xianyi/OpenBLAS/tree/arm_soft_fp_abi
            SET(OPENBLAS_COMMIT "b5c96fcfcdc82945502a2303116a64d89985daf5")
            SET(OPTIONAL_ARGS HOSTCC=${HOST_C_COMPILER} TARGET=ARMV7 ARM_SOFTFP_ABI=1 USE_THREAD=0)
        ELSEIF(RPI)
            # use hardfp
            SET(OPENBLAS_COMMIT "v0.2.19")
            SET(OPTIONAL_ARGS HOSTCC=${HOST_C_COMPILER} TARGET=ARMV7 USE_THREAD=0)
        ENDIF()
    ELSE()
        SET(OPENBLAS_COMMIT "v0.2.19")
        SET(OPTIONAL_ARGS "")
        IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^x86(_64)?$")
            SET(OPTIONAL_ARGS DYNAMIC_ARCH=1 NUM_THREADS=64)
        ENDIF()
    ENDIF()

    ExternalProject_Add(
        extern_openblas
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY      https://github.com/xianyi/OpenBLAS.git
        GIT_TAG             ${OPENBLAS_COMMIT}
        PREFIX              ${CBLAS_SOURCES_DIR}
        INSTALL_DIR         ${CBLAS_INSTALL_DIR}
        BUILD_IN_SOURCE     1
        BUILD_COMMAND       ${CMAKE_MAKE_PROGRAM} ${COMMON_ARGS} ${OPTIONAL_ARGS}
        INSTALL_COMMAND     ${CMAKE_MAKE_PROGRAM} install NO_SHARED=1 NO_LAPACK=1 PREFIX=<INSTALL_DIR>
        UPDATE_COMMAND      ""
        CONFIGURE_COMMAND   ""
    )
ENDIF(NOT ${CBLAS_FOUND})

MESSAGE(STATUS "BLAS library: ${CBLAS_LIBRARIES}")
INCLUDE_DIRECTORIES(${CBLAS_INC_DIR})

ADD_LIBRARY(cblas STATIC IMPORTED)
SET_PROPERTY(TARGET cblas PROPERTY IMPORTED_LOCATION ${CBLAS_LIBRARIES})
IF(NOT ${CBLAS_FOUND})
    ADD_DEPENDENCIES(cblas extern_openblas)
    LIST(APPEND external_project_dependencies cblas)
ENDIF(NOT ${CBLAS_FOUND})
