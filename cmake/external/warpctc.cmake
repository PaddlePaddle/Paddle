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

IF(MOBILE_INFERENCE)
    return()
ENDIF()

INCLUDE(ExternalProject)

SET(WARPCTC_SOURCES_DIR ${THIRD_PARTY_PATH}/warpctc)
SET(WARPCTC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/warpctc)

SET(WARPCTC_INCLUDE_DIR "${WARPCTC_INSTALL_DIR}/include"
    CACHE PATH "Warp-ctc Directory" FORCE)
# Used in unit test test_WarpCTCLayer
SET(WARPCTC_LIB_DIR "${WARPCTC_INSTALL_DIR}/lib"
    CACHE PATH "Warp-ctc Library Directory" FORCE)
SET(WARPCTC_LIBRARIES "${WARPCTC_INSTALL_DIR}/lib/libwarpctc${CMAKE_SHARED_LIBRARY_SUFFIX}"
    CACHE FILEPATH "Warp-ctc Library" FORCE)

IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )
    SET(USE_OMP OFF)
ELSE()
    SET(USE_OMP ON)
ENDIF()

ExternalProject_Add(
    extern_warpctc
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/gangliao/warp-ctc.git"
    GIT_TAG         b63a0644654a3e0ed624c85a1767bc8193aead09
    PREFIX          ${WARPCTC_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_INSTALL_PREFIX=${WARPCTC_INSTALL_DIR}
                    -DWITH_GPU=${WITH_GPU}
                    -DWITH_OMP=${USE_OMP}
                    -DWITH_TORCH=OFF
                    -DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON
                    -DBUILD_SHARED=ON
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_INSTALL_PREFIX:PATH=${WARPCTC_INSTALL_DIR}
)

MESSAGE(STATUS "warp-ctc library: ${WARPCTC_LIBRARIES}")
INCLUDE_DIRECTORIES(${WARPCTC_INCLUDE_DIR})

ADD_LIBRARY(warpctc STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET warpctc PROPERTY IMPORTED_LOCATION ${WARPCTC_LIBRARIES})
ADD_DEPENDENCIES(warpctc extern_warpctc)

LIST(APPEND external_project_dependencies warpctc)
