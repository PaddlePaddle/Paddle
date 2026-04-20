# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# Sleef external project configuration
# Sleef version: 3.6.1 (latest stable release)

set(SLEEF_REPOSITORY "https://github.com/shibatch/sleef.git")
set(SLEEF_TAG "3.6.1")

# Cache sleef source
cache_third_party(
  extern_sleef
  REPOSITORY
  ${SLEEF_REPOSITORY}
  TAG
  ${SLEEF_TAG}
  DIR
  SLEEF_SOURCE_DIR)

set(SLEEF_SOURCE_DIR
    "${SLEEF_SOURCE_DIR}"
    CACHE PATH "sleef source dir")
set(SLEEF_INSTALL_DIR "${THIRD_PARTY_PATH}/install/sleef")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  set(SLEEF_CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set(SLEEF_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
elseif(MSVC)
  set(SLEEF_CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  set(SLEEF_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set(SLEEF_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_FLAGS=${SLEEF_CMAKE_C_FLAGS}
    -DCMAKE_CXX_FLAGS=${SLEEF_CMAKE_CXX_FLAGS}
    -DCMAKE_INSTALL_PREFIX=${SLEEF_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR=${SLEEF_INSTALL_DIR}/lib
    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_DFT=OFF
    -DBUILD_QUAD=OFF
    -DENABLE_ALTDIV=OFF
    -DENABLE_ALTSQRT=OFF)

ExternalProject_Add(
  extern_sleef
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SLEEF_DOWNLOAD_CMD}
  PREFIX ${THIRD_PARTY_PATH}/sleef
  SOURCE_DIR ${SLEEF_SOURCE_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS ${SLEEF_CMAKE_ARGS}
  BUILD_BYPRODUCTS
    ${SLEEF_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}sleef${CMAKE_STATIC_LIBRARY_SUFFIX}
)

set(SLEEF_INCLUDE_DIR "${SLEEF_INSTALL_DIR}/include")
set(SLEEF_LIBRARIES
    "${SLEEF_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}sleef${CMAKE_STATIC_LIBRARY_SUFFIX}"
)

add_library(sleef STATIC IMPORTED GLOBAL)
set_target_properties(sleef PROPERTIES IMPORTED_LOCATION ${SLEEF_LIBRARIES})

add_dependencies(sleef extern_sleef)
include_directories(${SLEEF_INCLUDE_DIR})

set(SLEEF_FOUND TRUE)
