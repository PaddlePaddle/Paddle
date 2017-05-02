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

    # check fortran compiler and library
    IF(ANDROID)
        SET(OPENBLAS_COMMIT "b5c96fcfcdc82945502a2303116a64d89985daf5")
        SET(OPTIONAL_ARGS HOSTCC=${HOST_C_COMPILER} TARGET=ARMV7 ARM_SOFTFP_ABI=1 NOFORTRAN=1 USE_THREAD=0 libs)
    ELSEIF(RPI)
        SET(OPENBLAS_COMMIT "v0.2.19")
        SET(OPTIONAL_ARGS HOSTCC=${HOST_C_COMPILER} TARGET=ARMV7 NOFORTRAN=1 USE_THREAD=0 libs)
    ELSE()
        IF(CMAKE_COMPILER_IS_GNUCC)
            ENABLE_LANGUAGE(Fortran)
            if (NOT CMAKE_Fortran_COMPILER_VERSION)
              # cmake < 3.4 cannot get CMAKE_Fortran_COMPILER_VERSION directly.
              execute_process(COMMAND ${CMAKE_Fortran_COMPILER} -dumpversion
                        OUTPUT_VARIABLE CMAKE_Fortran_COMPILER_VERSION)
            endif()
            string(REGEX MATCHALL "[0-9]+" Fortran_VERSION ${CMAKE_Fortran_COMPILER_VERSION})
            list(GET Fortran_VERSION 0 Fortran_MAJOR)
            list(GET Fortran_VERSION 1 Fortran_MINOR)
            find_library(GFORTRAN_LIBRARY NAMES gfortran PATHS
                         /lib
                         /usr/lib
                         /usr/lib/gcc/x86_64-linux-gnu/${Fortran_MAJOR}.${Fortran_MINOR}/
                         /usr/lib/gcc/x86_64-linux-gnu/${Fortran_MAJOR}/)
            if (NOT GFORTRAN_LIBRARY)
                message(FATAL_ERROR "Cannot found gfortran library which it is used by openblas")
            endif()
            find_package(Threads REQUIRED)
            LIST(APPEND CBLAS_LIBRARIES ${GFORTRAN_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
        ENDIF(CMAKE_COMPILER_IS_GNUCC)

        IF(NOT CMAKE_Fortran_COMPILER)
            MESSAGE(FATAL_ERROR "To build lapack in libopenblas, "
                    "you need to set gfortran compiler: cmake .. -DCMAKE_Fortran_COMPILER=...")
        ENDIF(NOT CMAKE_Fortran_COMPILER)

        ADD_DEFINITIONS(-DPADDLE_USE_LAPACK)

        SET(OPENBLAS_COMMIT "v0.2.19")
        SET(OPENBLAS_ARGS FC=${CMAKE_Fortran_COMPILER} DYNAMIC_ARCH=1 libs netlib)
    ENDIF()

    ExternalProject_Add(
        openblas
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY      https://github.com/xianyi/OpenBLAS.git
        GIT_TAG             ${OPENBLAS_COMMIT}
        PREFIX              ${CBLAS_SOURCES_DIR}
        INSTALL_DIR         ${CBLAS_INSTALL_DIR}
        BUILD_IN_SOURCE     1
        BUILD_COMMAND       ${CMAKE_MAKE_PROGRAM} CC=${CMAKE_C_COMPILER} NO_SHARED=1 ${OPTIONAL_ARGS}
        INSTALL_COMMAND     ${CMAKE_MAKE_PROGRAM} install NO_SHARED=1 PREFIX=<INSTALL_DIR>
        UPDATE_COMMAND      ""
        CONFIGURE_COMMAND   ""
    )

    IF(NOT ANDROID AND NOT RPI)
        ExternalProject_Add_Step(
            openblas lapacke_install
            COMMAND ${CMAKE_COMMAND} -E copy "${CBLAS_SOURCES_DIR}/src/openblas/lapack-netlib/LAPACKE/include/lapacke_mangling_with_flags.h" "${CBLAS_INSTALL_DIR}/include/lapacke_mangling.h"
            COMMAND ${CMAKE_COMMAND} -E copy "${CBLAS_SOURCES_DIR}/src/openblas/lapack-netlib/LAPACKE/include/lapacke.h" "${CBLAS_INSTALL_DIR}/include/lapacke.h"
            COMMAND ${CMAKE_COMMAND} -E copy "${CBLAS_SOURCES_DIR}/src/openblas/lapack-netlib/LAPACKE/include/lapacke_config.h" "${CBLAS_INSTALL_DIR}/include/lapacke_config.h"
            COMMAND ${CMAKE_COMMAND} -E copy "${CBLAS_SOURCES_DIR}/src/openblas/lapack-netlib/LAPACKE/include/lapacke_utils.h" "${CBLAS_INSTALL_DIR}/include/lapacke_utils.h"
            DEPENDEES install
    )
    ENDIF()

    LIST(APPEND external_project_dependencies openblas)
ENDIF(NOT ${CBLAS_FOUND})

MESSAGE(STATUS "BLAS library: ${CBLAS_LIBRARIES}")
INCLUDE_DIRECTORIES(${CBLAS_INC_DIR})
