# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

include(ExternalProject)

set(LIBUV_PREFIX_DIR ${THIRD_PARTY_PATH}/libuv)
set(LIBUV_INCLUDE_DIR ${LIBUV_PREFIX_DIR}/include)

set(LIBUV_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/libuv)
set(SOURCE_INCLUDE_DIR ${LIBUV_SOURCE_DIR}/include)

set(LIBUV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/libuv)
set(LIBUV_INCLUDE_DIR
    ${LIBUV_INSTALL_DIR}/include
    CACHE PATH "libuv include directory." FORCE)
set(LIBUV_LIBRARY_DIR
    ${LIBUV_INSTALL_DIR}/lib
    CACHE PATH "libuv library directory." FORCE)

set(LIBUV_LIBRARIES
    ${LIBUV_INSTALL_DIR}/lib/libuv.a
    CACHE FILEPATH "libuv library." FORCE)

set(LIBUV_BuildTests
    OFF
    CACHE INTERNAL "")

set(LIBUV_CMAKE_C_FLAGS "-O0 -fPIC")
set(LIBUV_CMAKE_CXX_FLAGS "-O0 -fPIC")

ExternalProject_Add(
  extern_libuv
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${LIBUV_SOURCE_DIR}
  PREFIX ${LIBUV_PREFIX_DIR}
  PATCH_COMMAND ""
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
             -DCMAKE_INSTALL_PREFIX=${LIBUV_INSTALL_DIR}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_FLAGS=${LIBUV_CMAKE_C_FLAGS}
             -DCMAKE_CXX_FLAGS=${LIBUV_CMAKE_CXX_FLAGS}
  BUILD_BYPRODUCTS ${LIBUV_LIBRARIES})

add_library(libuv STATIC IMPORTED GLOBAL)
set_property(TARGET libuv PROPERTY IMPORTED_LOCATION ${LIBUV_LIBRARIES})
add_dependencies(libuv extern_libuv)

include_directories(${LIBUV_INCLUDE_DIR})

#add_library(libuv INTERFACE)
#set_property(TARGET libuv PROPERTY IMPORTED_LOCATION ${DGC_LIBRARIES})
