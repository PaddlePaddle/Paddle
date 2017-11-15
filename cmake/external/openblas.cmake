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

IF(USE_EIGEN_FOR_BLAS)
    return()
ENDIF(USE_EIGEN_FOR_BLAS)

INCLUDE(cblas)

IF(NOT ${CBLAS_FOUND})
    INCLUDE(ExternalProject)

    SET(CBLAS_SOURCES_DIR ${THIRD_PARTY_PATH}/openblas)
    SET(CBLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openblas)
    SET(CBLAS_INC_DIR "${CBLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)

    SET(CBLAS_LIBRARIES
        "${CBLAS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
        CACHE FILEPATH "openblas library." FORCE)

    SET(OPENBLAS_CC "${CMAKE_C_COMPILER}")

    IF(CMAKE_CROSSCOMPILING)
        SET(OPTIONAL_ARGS HOSTCC=${HOST_C_COMPILER})
        GET_FILENAME_COMPONENT(CROSS_SUFFIX ${CMAKE_C_COMPILER} DIRECTORY)
        SET(CROSS_SUFFIX ${CROSS_SUFFIX}/)
        IF(ANDROID)
            # arm_soft_fp_abi branch of OpenBLAS to support softfp
            #   https://github.com/xianyi/OpenBLAS/tree/arm_soft_fp_abi
            SET(OPENBLAS_COMMIT "b5c96fcfcdc82945502a2303116a64d89985daf5")
            IF(ANDROID_ABI MATCHES "^armeabi(-v7a)?$")
                SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} TARGET=ARMV7 ARM_SOFTFP_ABI=1 USE_THREAD=0)
            ELSEIF(ANDROID_ABI STREQUAL "arm64-v8a")
                SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} TARGET=ARMV8 BINARY=64 USE_THREAD=0)
            ENDIF()
        ELSEIF(IOS)
            # FIXME(liuyiqun): support multiple architectures
            SET(OPENBLAS_COMMIT "b5c96fcfcdc82945502a2303116a64d89985daf5")
            SET(OPENBLAS_CC "${OPENBLAS_CC} ${CMAKE_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
            IF(CMAKE_OSX_ARCHITECTURES MATCHES "armv7")
                SET(OPENBLAS_CC "${OPENBLAS_CC} -arch armv7")
                SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} TARGET=ARMV7 ARM_SOFTFP_ABI=1 USE_THREAD=0)
            ELSEIF(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
                SET(OPENBLAS_CC "${OPENBLAS_CC} -arch arm64")
                SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} TARGET=ARMV8 BINARY=64 USE_THREAD=0 CROSS_SUFFIX=${CROSS_SUFFIX})
            ENDIF()
        ELSEIF(RPI)
            # use hardfp
            SET(OPENBLAS_COMMIT "v0.2.20")
            SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} TARGET=ARMV7 USE_THREAD=0)
        ENDIF()
    ELSE()
        IF(APPLE)
            SET(OPENBLAS_CC "${CMAKE_C_COMPILER} -isysroot ${CMAKE_OSX_SYSROOT}")
        ENDIF()
        SET(OPENBLAS_COMMIT "v0.2.20")
        SET(OPTIONAL_ARGS "")
        IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^x86(_64)?$")
            SET(OPTIONAL_ARGS DYNAMIC_ARCH=1 NUM_THREADS=64)
        ENDIF()
    ENDIF()

    SET(COMMON_ARGS CC=${OPENBLAS_CC} NO_SHARED=1 NO_LAPACK=1 libs)

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
    SET(CBLAS_PROVIDER openblas)
    IF(WITH_C_API)
        INSTALL(DIRECTORY ${CBLAS_INC_DIR} DESTINATION third_party/openblas)
        # Because libopenblas.a is a symbolic link of another library, thus need to
        # install the whole directory.
        IF(ANDROID)
            SET(TMP_INSTALL_DIR third_party/openblas/lib/${ANDROID_ABI})
        ELSE()
            SET(TMP_INSTALL_DIR third_party/openblas/lib)
        ENDIF()
        INSTALL(CODE "execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CBLAS_INSTALL_DIR}/lib
                    destination ${CMAKE_INSTALL_PREFIX}/${TMP_INSTALL_DIR}
            )"
        )
        INSTALL(CODE "MESSAGE(STATUS \"Installing: \"
                \"${CBLAS_INSTALL_DIR}/lib -> ${CMAKE_INSTALL_PREFIX}/${TMP_INSTALL_DIR}\"
            )"
        )
    ENDIF()
ENDIF(NOT ${CBLAS_FOUND})

MESSAGE(STATUS "BLAS library: ${CBLAS_LIBRARIES}")
INCLUDE_DIRECTORIES(${CBLAS_INC_DIR})

# FIXME(gangliao): generate cblas target to track all high performance
# linear algebra libraries for cc_library(xxx SRCS xxx.c DEPS cblas)
SET(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/cblas_dummy.c)
FILE(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")
IF("${CBLAS_PROVIDER}" STREQUAL "MKLML")
    ADD_LIBRARY(cblas SHARED ${dummyfile})
ELSE()
    ADD_LIBRARY(cblas STATIC ${dummyfile})
ENDIF()
TARGET_LINK_LIBRARIES(cblas ${CBLAS_LIBRARIES})

IF(NOT ${CBLAS_FOUND})
    ADD_DEPENDENCIES(cblas extern_openblas)
    LIST(APPEND external_project_dependencies cblas)
ELSE()
    IF("${CBLAS_PROVIDER}" STREQUAL "MKLML")
        ADD_DEPENDENCIES(cblas mklml)
    ENDIF()
ENDIF(NOT ${CBLAS_FOUND})
