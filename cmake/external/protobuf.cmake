# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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

SET(PROTOBUF_SOURCES_DIR ${THIRD_PARTY_PATH}/protobuf)
SET(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/protobuf)
SET(PROTOBUF_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" CACHE PATH "protobuf include directory." FORCE)

INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

IF(WIN32)
  SET(PROTOBUF_LITE_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite.lib" CACHE FILEPATH "protobuf lite library." FORCE)
  SET(PROTOBUF_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf.lib" CACHE FILEPATH "protobuf library." FORCE)
  SET(PROTOBUF_PROTOC_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotoc.lib" CACHE FILEPATH "protoc library." FORCE)
  SET(PROTOBUF_PROTOC_EXECUTABLE "${PROTOBUF_INSTALL_DIR}/bin/protoc.exe" CACHE FILEPATH "protobuf executable." FORCE)
ELSE(WIN32)
  SET(PROTOBUF_LITE_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite.a" CACHE FILEPATH "protobuf lite library." FORCE)
  SET(PROTOBUF_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf.a" CACHE FILEPATH "protobuf library." FORCE)
  SET(PROTOBUF_PROTOC_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotoc.a" CACHE FILEPATH "protoc library." FORCE)
  SET(PROTOBUF_PROTOC_EXECUTABLE "${PROTOBUF_INSTALL_DIR}/bin/protoc" CACHE FILEPATH "protobuf executable." FORCE)
ENDIF(WIN32)

ExternalProject_Add(
  protobuf
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX          ${PROTOBUF_SOURCES_DIR}
  UPDATE_COMMAND  ""
  DEPENDS         zlib
  GIT_REPOSITORY  "https://github.com/google/protobuf.git"
  GIT_TAG         "9f75c5aa851cd877fb0d93ccc31b8567a6706546"
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} ${PROTOBUF_SOURCES_DIR}/src/protobuf/cmake
    -Dprotobuf_BUILD_TESTS=OFF
    -DZLIB_ROOT:FILEPATH=${ZLIB_ROOT}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR=lib
)

LIST(APPEND external_project_dependencies protobuf)
