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

set(ROCKSDB_PREFIX_DIR ${THIRD_PARTY_PATH}/rocksdb)
set(ROCKSDB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/rocksdb)
set(ROCKSDB_INCLUDE_DIR
    "${ROCKSDB_INSTALL_DIR}/include"
    CACHE PATH "rocksdb include directory." FORCE)
set(ROCKSDB_LIBRARIES
    "${ROCKSDB_INSTALL_DIR}/lib/librocksdb.a"
    CACHE FILEPATH "rocksdb library." FORCE)
set(ROCKSDB_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
include_directories(${ROCKSDB_INCLUDE_DIR})

ExternalProject_Add(
  extern_rocksdb
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${ROCKSDB_PREFIX_DIR}
  GIT_REPOSITORY "https://github.com/facebook/rocksdb"
  GIT_TAG v6.10.1
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DWITH_BZ2=OFF
             -DPORTABLE=1
             -DWITH_GFLAGS=OFF
             -DCMAKE_CXX_FLAGS=${ROCKSDB_CMAKE_CXX_FLAGS}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  #    BUILD_BYPRODUCTS ${ROCKSDB_PREFIX_DIR}/src/extern_rocksdb/librocksdb.a
  INSTALL_COMMAND
    mkdir -p ${ROCKSDB_INSTALL_DIR}/lib/ && cp
    ${ROCKSDB_PREFIX_DIR}/src/extern_rocksdb/librocksdb.a ${ROCKSDB_LIBRARIES}
    && cp -r ${ROCKSDB_PREFIX_DIR}/src/extern_rocksdb/include
    ${ROCKSDB_INSTALL_DIR}/
  BUILD_IN_SOURCE 1
  BYPRODUCTS ${ROCKSDB_LIBRARIES})

add_dependencies(extern_rocksdb snappy)

add_library(rocksdb STATIC IMPORTED GLOBAL)
set_property(TARGET rocksdb PROPERTY IMPORTED_LOCATION ${ROCKSDB_LIBRARIES})
add_dependencies(rocksdb extern_rocksdb)

list(APPEND external_project_dependencies rocksdb)
