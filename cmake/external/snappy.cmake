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

include (ExternalProject)

# NOTE: snappy is needed when linking with recordio

set(SNAPPY_PREFIX_DIR ${THIRD_PARTY_PATH}/snappy)
set(SNAPPY_INSTALL_DIR ${THIRD_PARTY_PATH}/install/snappy)
set(SNAPPY_INCLUDE_DIR "${SNAPPY_INSTALL_DIR}/include" CACHE PATH "snappy include directory." FORCE)

if(WIN32)
    SET(SNAPPY_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267")
    IF(NOT EXISTS "${SNAPPY_INSTALL_DIR}/lib/libsnappy.lib")
        add_custom_command(TARGET extern_snappy POST_BUILD
                COMMAND cmake -E copy ${SNAPPY_INSTALL_DIR}/lib/snappy.lib ${SNAPPY_INSTALL_DIR}/lib/libsnappy.lib
                )
    ENDIF()
    set(SNAPPY_LIBRARIES "${SNAPPY_INSTALL_DIR}/lib/libsnappy.lib")
else()
    SET(SNAPPY_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(SNAPPY_LIBRARIES "${SNAPPY_INSTALL_DIR}/lib/libsnappy.a")
endif()

ExternalProject_Add(
        extern_snappy
        GIT_REPOSITORY "https://github.com/google/snappy"
        GIT_TAG "1.1.7"
        PREFIX          ${SNAPPY_PREFIX_DIR}
        UPDATE_COMMAND  ""
        CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                        -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                        -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                        -DCMAKE_CXX_FLAGS=${SNAPPY_CMAKE_CXX_FLAGS}
                        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                        -DCMAKE_INSTALL_PREFIX=${SNAPPY_INSTALL_DIR}
                        -DCMAKE_INSTALL_LIBDIR=${SNAPPY_INSTALL_DIR}/lib
                        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                        -DBUILD_TESTING=OFF
                        -DSNAPPY_BUILD_TESTS:BOOL=OFF
                        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                        ${EXTERNAL_OPTIONAL_ARGS}
        CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${SNAPPY_INSTALL_DIR}
                         -DCMAKE_INSTALL_LIBDIR:PATH=${SNAPPY_INSTALL_DIR}/lib
                         -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                         -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
        BUILD_BYPRODUCTS ${SNAPPY_LIBRARIES}
)

add_library(snappy STATIC IMPORTED GLOBAL)
set_property(TARGET snappy PROPERTY IMPORTED_LOCATION ${SNAPPY_LIBRARIES})

include_directories(${SNAPPY_INCLUDE_DIR})
add_dependencies(snappy extern_snappy)

