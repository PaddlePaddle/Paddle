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

set(LIBUV_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/libuv)
set(LIBUV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/libuv)

if(WIN32)
  set(LIBUV_LIBRARIES ${LIBUV_INSTALL_DIR}/lib/libuv.lib)
  set(LIBUV_INCLUDE_DIR ${LIBUV_INSTALL_DIR}/include)
else()
  # Unix-like platform (Linux or macOS)
  set(LIBUV_LIBRARIES ${LIBUV_INSTALL_DIR}/lib/libuv.a)
  set(LIBUV_INCLUDE_DIR ${LIBUV_INSTALL_DIR}/include)
endif()

ExternalProject_Add(
  extern_libuv
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${LIBUV_SOURCE_DIR}
  BINARY_DIR ${LIBUV_SOURCE_DIR}
  INSTALL_DIR ${LIBUV_INSTALL_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${LIBUV_INSTALL_DIR}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
             -DBUILD_STATIC_LIBS=ON
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DBUILD_TESTING=OFF
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${LIBUV_INSTALL_DIR}
  # output
  BUILD_BYPRODUCTS ${LIBUV_LIBRARIES})

add_library(libuv STATIC IMPORTED)
add_dependencies(libuv extern_libuv)

set_target_properties(libuv PROPERTIES IMPORTED_LOCATION ${LIBUV_LIBRARIES})
if(WIN32)
  set_target_properties(
    libuv PROPERTIES INTERFACE_LINK_LIBRARIES
                     "ws2_32;psapi;iphlpapi;userenv;advapi32")
endif()

include_directories(${LIBUV_INCLUDE_DIR})
