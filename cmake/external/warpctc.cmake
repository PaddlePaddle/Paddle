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

INCLUDE(ExternalProject)

SET(WARPCTC_SOURCES_DIR ${THIRD_PARTY_PATH}/warpctc)
SET(WARPCTC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/warpctc)

SET(WARPCTC_INCLUDE_DIR "${WARPCTC_INSTALL_DIR}/include"
    CACHE PATH "Warp-ctc Directory" FORCE)
# Used in unit test test_WarpCTCLayer
SET(WARPCTC_LIB_DIR "${WARPCTC_INSTALL_DIR}/lib"
    CACHE PATH "Warp-ctc Library Directory" FORCE)

IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR WIN32)
    SET(USE_OMP OFF)
ELSE()
    SET(USE_OMP ON)
ENDIF()

IF(WIN32)
    SET(WARPCTC_REPOSITORY "https://github.com/wopeizl/warp-ctc.git")
ELSE()
    SET(WARPCTC_REPOSITORY "https://github.com/dzhwinter/warp-ctc.git")
ENDIF()

ExternalProject_Add(
    extern_warpctc
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY ${WARPCTC_REPOSITORY}
    PREFIX          ${WARPCTC_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_INSTALL_PREFIX=${WARPCTC_INSTALL_DIR}
                    -DWITH_GPU=${WITH_GPU}
                    -DWITH_OMP=${USE_OMP}
                    -DWITH_TORCH=OFF
                    -DCMAKE_DISABLE_FIND_PACKAGE_Torch=ON
                    -DBUILD_SHARED=ON
                    -DBUILD_TESTS=OFF
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_INSTALL_PREFIX:PATH=${WARPCTC_INSTALL_DIR}
)
IF(WIN32)
    IF(NOT EXISTS "${WARPCTC_INSTALL_DIR}/lib/warpctc${CMAKE_SHARED_LIBRARY_SUFFIX}")
        add_custom_command(TARGET extern_warpctc POST_BUILD
                COMMAND cmake -E copy ${WARPCTC_INSTALL_DIR}/bin/warpctc${CMAKE_SHARED_LIBRARY_SUFFIX} ${WARPCTC_INSTALL_DIR}/lib/warpctc${CMAKE_SHARED_LIBRARY_SUFFIX}
                )
    ENDIF()
    SET(WARPCTC_LIBRARIES "${WARPCTC_INSTALL_DIR}/lib/warpctc${CMAKE_SHARED_LIBRARY_SUFFIX}"
            CACHE FILEPATH "Warp-ctc Library" FORCE)
else(WIN32)
    SET(WARPCTC_LIBRARIES "${WARPCTC_INSTALL_DIR}/lib/libwarpctc${CMAKE_SHARED_LIBRARY_SUFFIX}"
            CACHE FILEPATH "Warp-ctc Library" FORCE)
ENDIF(WIN32)

MESSAGE(STATUS "warp-ctc library: ${WARPCTC_LIBRARIES}")
get_filename_component(WARPCTC_LIBRARY_PATH ${WARPCTC_LIBRARIES} DIRECTORY)
INCLUDE_DIRECTORIES(${WARPCTC_INCLUDE_DIR}) # For warpctc code to include its headers.
INCLUDE_DIRECTORIES(${THIRD_PARTY_PATH}/install) # For Paddle code to include warpctc headers.

ADD_LIBRARY(warpctc SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET warpctc PROPERTY IMPORTED_LOCATION ${WARPCTC_LIBRARIES})
ADD_DEPENDENCIES(warpctc extern_warpctc)
