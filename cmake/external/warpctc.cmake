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

SET(WARPCTC_SOURCES_DIR ${THIRD_PARTY_PATH}/warpctc)
SET(WARPCTC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/warpctc)
SET(WARPCTC_INCLUDE_DIR "${WARPCTC_INSTALL_DIR}/include" CACHE PATH "Warp-ctc Directory" FORCE)

INCLUDE_DIRECTORIES(${WARPCTC_INCLUDE_DIR})

SET(WARPCTC_LIB_DIR "${WARPCTC_INSTALL_DIR}/lib" CACHE PATH "Warp-ctc Library Directory" FORCE)

IF(WIN32)
    SET(WARPCTC_LIBRARIES
        "${WARPCTC_INSTALL_DIR}/lib/warpctc.dll" CACHE FILEPATH "Warp-ctc Library" FORCE)
ELSE(WIN32)
    IF(APPLE)
        SET(_warpctc_SHARED_SUFFIX dylib)
    ELSE(APPLE)
        SET(_warpctc_SHARED_SUFFIX so)
    ENDIF(APPLE)

    SET(WARPCTC_LIBRARIES
        "${WARPCTC_INSTALL_DIR}/lib/libwarpctc.${_warpctc_SHARED_SUFFIX}" CACHE FILEPATH "Warp-ctc Library" FORCE)
ENDIF(WIN32)

IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )
    SET(USE_OMP OFF)
ELSE()
    SET(USE_OMP ON)
ENDIF()

ExternalProject_Add(
    warpctc
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/gangliao/warp-ctc.git"
    PREFIX          ${WARPCTC_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    CMAKE_ARGS      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    CMAKE_ARGS      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    CMAKE_ARGS      -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
    CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${WARPCTC_INSTALL_DIR}
    CMAKE_ARGS      -DWITH_GPU=${WITH_GPU}
    CMAKE_ARGS      -DWITH_OMP=${USE_OMP}
    CMAKE_ARGS      -DWITH_TORCH=OFF
    CMAKE_ARGS      -DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON
    CMAKE_ARGS      -DBUILD_SHARED=ON
    CMAKE_ARGS      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
    CMAKE_CACHE_ARGS -DCMAKE_BUILD_TYPE:STRING=Release
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_INSTALL_PREFIX:PATH=${WARPCTC_INSTALL_DIR}
)

LIST(APPEND external_project_dependencies warpctc)
