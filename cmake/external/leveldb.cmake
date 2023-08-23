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

set(LEVELDB_TAG v1.18)
set(LEVELDB_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/leveldb)
set(LEVELDB_PREFIX_DIR ${THIRD_PARTY_PATH}/leveldb)
set(LEVELDB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/leveldb)
set(LEVELDB_INCLUDE_DIR
    "${LEVELDB_INSTALL_DIR}/include"
    CACHE PATH "leveldb include directory." FORCE)
set(LEVELDB_LIBRARIES
    "${LEVELDB_INSTALL_DIR}/lib/libleveldb.a"
    CACHE FILEPATH "leveldb library." FORCE)
include_directories(${LEVELDB_INCLUDE_DIR})
set(LEVELDN_CXXFLAGS "-fPIC")
if(WITH_HETERPS AND WITH_PSLIB)
  set(LEVELDN_CXXFLAGS "${LEVELDN_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

ExternalProject_Add(
  extern_leveldb
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${LEVELDB_PREFIX_DIR}
  SOURCE_DIR ${LEVELDB_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND export "CXXFLAGS=${LEVELDN_CXXFLAGS}" && make -j
                ${NUM_OF_PROCESSOR} libleveldb.a
  INSTALL_COMMAND
    mkdir -p ${LEVELDB_INSTALL_DIR}/lib/ && cp
    ${LEVELDB_SOURCE_DIR}/libleveldb.a ${LEVELDB_LIBRARIES} && cp -r
    ${LEVELDB_SOURCE_DIR}/include ${LEVELDB_INSTALL_DIR}/
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS ${LEVELDB_LIBRARIES})
add_dependencies(extern_leveldb snappy)

add_library(leveldb STATIC IMPORTED GLOBAL)
set_property(TARGET leveldb PROPERTY IMPORTED_LOCATION ${LEVELDB_LIBRARIES})
add_dependencies(leveldb extern_leveldb)

list(APPEND external_project_dependencies leveldb)
