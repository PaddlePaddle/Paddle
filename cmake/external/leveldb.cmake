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

SET(LEVELDB_PREFIX_DIR ${THIRD_PARTY_PATH}/leveldb)
SET(LEVELDB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/leveldb)
SET(LEVELDB_INCLUDE_DIR "${LEVELDB_INSTALL_DIR}/include" CACHE PATH "leveldb include directory." FORCE)
SET(LEVELDB_LIBRARIES "${LEVELDB_INSTALL_DIR}/lib/libleveldb.a" CACHE FILEPATH "leveldb library." FORCE)
INCLUDE_DIRECTORIES(${LEVELDB_INCLUDE_DIR})

ExternalProject_Add(
        extern_leveldb
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX ${LEVELDB_PREFIX_DIR}
        GIT_REPOSITORY "https://github.com/google/leveldb"
        GIT_TAG v1.18
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND CXXFLAGS=-fPIC make -j ${NUM_OF_PROCESSOR} libleveldb.a
        INSTALL_COMMAND mkdir -p ${LEVELDB_INSTALL_DIR}/lib/
        && cp ${LEVELDB_PREFIX_DIR}/src/extern_leveldb/libleveldb.a ${LEVELDB_LIBRARIES}
        && cp -r ${LEVELDB_PREFIX_DIR}/src/extern_leveldb/include ${LEVELDB_INSTALL_DIR}/
        BUILD_IN_SOURCE 1
        BUILD_BYPRODUCTS ${LEVELDB_LIBRARIES}
)

ADD_DEPENDENCIES(extern_leveldb snappy)

ADD_LIBRARY(leveldb STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET leveldb PROPERTY IMPORTED_LOCATION ${LEVELDB_LIBRARIES})
ADD_DEPENDENCIES(leveldb extern_leveldb)

LIST(APPEND external_project_dependencies leveldb)

