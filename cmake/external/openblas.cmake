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
INCLUDE(cblas)

IF(NOT ${CBLAS_FOUND})
    INCLUDE(ExternalProject)

    SET(CBLAS_SOURCES_DIR ${THIRD_PARTY_PATH}/openblas)
    SET(CBLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openblas)
    SET(CBLAS_INC_DIR "${CBLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)

    SET(CBLAS_LIBRARIES
        "${CBLAS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
        CACHE FILEPATH "openblas library." FORCE)

    ADD_DEFINITIONS(-DPADDLE_USE_OPENBLAS)

    IF (WIN32)
        SET(CBLAS_FOUND true)
        MESSAGE(WARNING, "In windows, openblas only support msvc build, please build it manually and put it at " ${CBLAS_INSTALL_DIR})
    ENDIF(WIN32)

    IF (NOT WIN32)
    SET(OPENBLAS_CC "${CMAKE_C_COMPILER} -Wno-unused-but-set-variable -Wno-unused-variable")
    SET(OPENBLAS_COMMIT "v0.2.20")

    IF(APPLE)
        SET(OPENBLAS_CC "${CMAKE_C_COMPILER} -isysroot ${CMAKE_OSX_SYSROOT}")
    ENDIF()
    SET(OPTIONAL_ARGS "")
    IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^x86(_64)?$")
        SET(OPTIONAL_ARGS DYNAMIC_ARCH=1 NUM_THREADS=64)
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
                            && rm -r ${CBLAS_INSTALL_DIR}/lib/cmake ${CBLAS_INSTALL_DIR}/lib/pkgconfig
        UPDATE_COMMAND      ""
        CONFIGURE_COMMAND   ""
    )
    ELSE()
    ENDIF(NOT WIN32)
    SET(CBLAS_PROVIDER openblas)
ENDIF(NOT ${CBLAS_FOUND})

MESSAGE(STATUS "BLAS library: ${CBLAS_LIBRARIES}")
MESSAGE(STATUS "BLAS Include: ${CBLAS_INC_DIR}")
INCLUDE_DIRECTORIES(${CBLAS_INC_DIR})

# FIXME(gangliao): generate cblas target to track all high performance
# linear algebra libraries for cc_library(xxx SRCS xxx.c DEPS cblas)
SET(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/cblas_dummy.c)
FILE(WRITE ${dummyfile} "const char *dummy_cblas = \"${dummyfile}\";")
ADD_LIBRARY(cblas STATIC ${dummyfile})

IF("${CBLAS_PROVIDER}" STREQUAL "MKLML")
  TARGET_LINK_LIBRARIES(cblas dynload_mklml)
ELSE()
  TARGET_LINK_LIBRARIES(cblas ${CBLAS_LIBRARIES})
ENDIF("${CBLAS_PROVIDER}" STREQUAL "MKLML")

IF(WITH_LIBXSMM)
  TARGET_LINK_LIBRARIES(cblas ${LIBXSMM_LIBS})
  ADD_DEPENDENCIES(cblas extern_libxsmm)
ENDIF()

IF(NOT ${CBLAS_FOUND})
    ADD_DEPENDENCIES(cblas extern_openblas)
ELSE()
    IF("${CBLAS_PROVIDER}" STREQUAL "MKLML")
        ADD_DEPENDENCIES(cblas mklml)
    ENDIF()
ENDIF(NOT ${CBLAS_FOUND})
