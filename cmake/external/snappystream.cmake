# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

IF(MOBILE_INFERENCE OR RPI)
    return()
ENDIF()

include (ExternalProject)

# NOTE: snappy is needed when linking with recordio

set(SNAPPYSTREAM_SOURCES_DIR ${THIRD_PARTY_PATH}/snappy_stream)
set(SNAPPYSTREAM_INSTALL_DIR ${THIRD_PARTY_PATH}/install/snappy_stream)
set(SNAPPYSTREAM_INCLUDE_DIR "${SNAPPYSTREAM_INSTALL_DIR}/include" CACHE PATH "snappy stream include directory." FORCE)

set(SNAPPYSTREAM_LIBRARIES "${SNAPPYSTREAM_INSTALL_DIR}/lib/libsnappystream.a")

ExternalProject_Add(
        extern_snappystream
        GIT_REPOSITORY "https://github.com/hoxnox/snappystream.git"
        GIT_TAG "0.2.8"
        PREFIX          ${SNAPPYSTREAM_SOURCES_DIR}
        UPDATE_COMMAND  ""
        CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                        -DCMAKE_INSTALL_PREFIX=${SNAPPY_INSTALL_DIR}
                        -DCMAKE_INSTALL_LIBDIR=${SNAPPY_INSTALL_DIR}/lib
                        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                        -DSNAPPY_ROOT=${SNAPPY_INSTALL_DIR}
                        ${EXTERNAL_OPTIONAL_ARGS}
                        CMAKE_CACHE_ARGS
                        -DCMAKE_INSTALL_PREFIX:PATH=${SNAPPYSTREAM_INSTALL_DIR}
                        -DCMAKE_INSTALL_LIBDIR:PATH=${SNAPPYSTREAM_INSTALL_DIR}/lib
                        -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
        BUILD_COMMAND   make -j8
        INSTALL_COMMAND make install
        DEPENDS snappy
)

add_library(snappystream STATIC IMPORTED GLOBAL)
set_property(TARGET snappystream PROPERTY IMPORTED_LOCATION ${SNAPPYSTREAM_LIBRARIES})

include_directories(${SNAPPYSTREAM_INCLUDE_DIR}) # For snappysteam to include its own headers.
include_directories(${THIRD_PARTY_PATH}/install) # For Paddle to include snappy stream headers.

add_dependencies(snappystream extern_snappystream)
