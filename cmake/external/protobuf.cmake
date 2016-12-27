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

SET(PROTOBUF_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf)
SET(PROTOBUF_INSTALL_DIR ${PROJECT_BINARY_DIR}/protobuf)

ExternalProject_Add(
    protobuf
    PREFIX          ${PROTOBUF_SOURCES_DIR}
    DEPENDS         zlib
    GIT_REPOSITORY  "https://github.com/google/protobuf.git"
#   GIT_TAG         "v3.1.0"
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} ${PROTOBUF_SOURCES_DIR}/src/protobuf/cmake
        -Dprotobuf_BUILD_TESTS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
    UPDATE_COMMAND ""
)

SET(PROTOBUF_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" CACHE PATH "protobuf include directory." FORCE)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

IF(WIN32)
  SET(PROTOBUF_LIBRARIES
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite.lib"
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf.lib"
        "${PROTOBUF_INSTALL_DIR}/lib/libprotoc.lib" CACHE FILEPATH "protobuf libraries." FORCE)
  SET(PROTOBUF_PROTOC_EXECUTABLE "${PROTOBUF_INSTALL_DIR}/bin/protoc.exe" CACHE FILEPATH "protobuf executable." FORCE)
ELSE(WIN32)
  SET(PROTOBUF_LIBRARIES
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite.a"
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf.a"
        "${PROTOBUF_INSTALL_DIR}/lib/libprotoc.a" CACHE FILEPATH "protobuf libraries." FORCE)
  SET(PROTOBUF_PROTOC_EXECUTABLE "${PROTOBUF_INSTALL_DIR}/bin/protoc" CACHE FILEPATH "protobuf executable." FORCE)
ENDIF(WIN32)

LIST(APPEND external_project_dependencies protobuf)
