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

include(ExternalProject)
if(NOT WITH_ARM)
  set(OPENSSL_USE_STATIC_LIBS ON)
endif()
find_package(OpenSSL REQUIRED)

message(STATUS "ssl:" ${OPENSSL_SSL_LIBRARY})
message(STATUS "crypto:" ${OPENSSL_CRYPTO_LIBRARY})

add_library(ssl SHARED IMPORTED GLOBAL)
set_property(TARGET ssl PROPERTY IMPORTED_LOCATION ${OPENSSL_SSL_LIBRARY})

add_library(crypto SHARED IMPORTED GLOBAL)
set_property(TARGET crypto PROPERTY IMPORTED_LOCATION ${OPENSSL_CRYPTO_LIBRARY})

set(BRPC_PREFIX_DIR ${THIRD_PARTY_PATH}/brpc)
set(BRPC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/brpc)
set(BRPC_INCLUDE_DIR
    "${BRPC_INSTALL_DIR}/include"
    CACHE PATH "brpc include directory." FORCE)
set(BRPC_LIBRARIES
    "${BRPC_INSTALL_DIR}/lib/libbrpc.a"
    CACHE FILEPATH "brpc library." FORCE)

include_directories(${BRPC_INCLUDE_DIR})

# clone brpc to Paddle/third_party
set(BRPC_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/brpc)
set(BRPC_URL ${GIT_URL}/apache/brpc.git)
set(BRPC_TAG 1.4.0)

# Reference https://stackoverflow.com/questions/45414507/pass-a-list-of-prefix-paths-to-externalproject-add-in-cmake-args
set(prefix_path
    "${THIRD_PARTY_PATH}/install/gflags|${THIRD_PARTY_PATH}/install/leveldb|${THIRD_PARTY_PATH}/install/snappy|${THIRD_PARTY_PATH}/install/gtest|${THIRD_PARTY_PATH}/install/protobuf|${THIRD_PARTY_PATH}/install/zlib|${THIRD_PARTY_PATH}/install/glog"
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION}
                                            VERSION_GREATER_EQUAL 13.0)
  file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/brpc/http2.h.patch
       http2_h_patch)
  set(BRPC_PATCH_COMMAND_GCC13 git apply ${http2_h_patch})
endif()

# If minimal .a is need, you can set  WITH_DEBUG_SYMBOLS=OFF
ExternalProject_Add(
  extern_brpc
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${BRPC_SOURCE_DIR}
  PREFIX ${BRPC_PREFIX_DIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND
  COMMAND git checkout -- . && git checkout ${BRPC_TAG}
  COMMAND ${BRPC_PATCH_COMMAND_GCC13}
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_INSTALL_PREFIX=${BRPC_INSTALL_DIR}
             -DCMAKE_INSTALL_LIBDIR=${BRPC_INSTALL_DIR}/lib
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             -DCMAKE_PREFIX_PATH=${prefix_path}
             -DWITH_GLOG=ON
             -DBUILD_BRPC_TOOLS=ON
             -DBUILD_SHARED_LIBS=ON
             ${EXTERNAL_OPTIONAL_ARGS}
  LIST_SEPARATOR |
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${BRPC_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR:PATH=${BRPC_INSTALL_DIR}/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${BRPC_LIBRARIES})

# ADD_DEPENDENCIES(extern_brpc protobuf ssl crypto leveldb gflags glog gtest snappy)
add_dependencies(
  extern_brpc
  protobuf
  ssl
  crypto
  leveldb
  gflags
  glog
  snappy)
add_library(brpc STATIC IMPORTED GLOBAL)
set_property(TARGET brpc PROPERTY IMPORTED_LOCATION ${BRPC_LIBRARIES})
add_dependencies(brpc extern_brpc)

add_definitions(-DBRPC_WITH_GLOG)

list(APPEND external_project_dependencies brpc)

set(EXTERNAL_BRPC_DEPS
    brpc
    protobuf
    ssl
    crypto
    leveldb
    glog
    snappy)

if(NOT WITH_GFLAGS)
  set(EXTERNAL_BRPC_DEPS ${EXTERNAL_BRPC_DEPS} gflags)
endif()
