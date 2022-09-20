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

# Reference https://stackoverflow.com/questions/45414507/pass-a-list-of-prefix-paths-to-externalproject-add-in-cmake-args
set(prefix_path
    "${THIRD_PARTY_PATH}/install/gflags|${THIRD_PARTY_PATH}/install/leveldb|${THIRD_PARTY_PATH}/install/snappy|${THIRD_PARTY_PATH}/install/gtest|${THIRD_PARTY_PATH}/install/protobuf|${THIRD_PARTY_PATH}/install/zlib|${THIRD_PARTY_PATH}/install/glog"
)

# If minimal .a is need, you can set  WITH_DEBUG_SYMBOLS=OFF
ExternalProject_Add(
  extern_brpc
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY "https://github.com/apache/incubator-brpc"
  GIT_TAG 1.2.0
  PREFIX ${BRPC_PREFIX_DIR}
  UPDATE_COMMAND ""
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
