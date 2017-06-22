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
    SET(PROTOBUF_LIBRARIES "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf.lib" CACHE FILEPATH "protobuf library." FORCE)
ELSE(WIN32)
    SET(PROTOBUF_LIBRARIES "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf.a" CACHE FILEPATH "protobuf library." FORCE)
ENDIF(WIN32)

ExternalProject_Add(
  extern_protobuf
  ${EXTERNAL_PROJECT_LOG_ARGS}
  DEPENDS         zlib
  GIT_REPOSITORY  "https://github.com/google/protobuf.git"
  GIT_TAG         "v3.1.0"
  PREFIX          ${PROTOBUF_SOURCES_DIR}
  UPDATE_COMMAND  ""
  CONFIGURE_COMMAND
  ${CMAKE_COMMAND} ${PROTOBUF_SOURCES_DIR}/src/protobuf/cmake
  -Dprotobuf_BUILD_TESTS=OFF
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
  -DCMAKE_INSTALL_LIBDIR=lib
  CMAKE_CACHE_ARGS
  -DCMAKE_INSTALL_PREFIX:PATH=${PROTOBUF_INSTALL_DIR}
  -DCMAKE_BUILD_TYPE:STRING=Release
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
  -DZLIB_ROOT:STRING=${ZLIB_ROOT}
  )

ADD_LIBRARY(protobuf STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET protobuf PROPERTY IMPORTED_LOCATION ${PROTOBUF_LIBRARIES})
ADD_DEPENDENCIES(protobuf extern_protobuf)

LIST(APPEND external_project_dependencies protobuf)

# By calling find_package, we got CMake function protobuf_generate_cpp.
find_package(Protobuf ${PROTOBUF_VERSION})

# Overwrite varaibles supposed to be defined by find_pacakge(Protobuf):
set(PROTOBUF_FOUND ON)
set(PROTOBUF_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIR})

IF(WIN32)
    SET(PROTOBUF_PROTOC_LIBRARIES "${PROTOBUF_INSTALL_DIR}/lib/libprotoc.lib" CACHE FILEPATH "protoc library." FORCE)
ELSE(WIN32)
    SET(PROTOBUF_PROTOC_LIBRARIES "${PROTOBUF_INSTALL_DIR}/lib/libprotoc.a" CACHE FILEPATH "protoc library." FORCE)
ENDIF(WIN32)

IF(WIN32)
    SET(PROTOBUF_LITE_LIBRARIES "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite.lib" CACHE FILEPATH "protobuf-lite library." FORCE)
ELSE(WIN32)
    SET(PROTOBUF_LITE_LIBRARIES "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf_lite.a" CACHE FILEPATH "protobuf-lite library." FORCE)
ENDIF(WIN32)

set(PROTOBUF_LIBRARY ${PROTOBUF_LIBRARIES})
set(PROTOBUF_PROTOC_LIBRARY ${PROTOBUF_LITE_LIBRARIES})

set(PROTOBUF_PROTOC_EXECUTABLE "${PROTOBUF_INSTALL_DIR}/bin/protoc")

set(PROTOBUF_LIBRARY_DEBUG ${PROTOBUF_LIBRARY})
set(PROTOBUF_PROTOC_LIBRARY_DEBUG ${PROTOBUF_PROTOC_LIBRARY})
set(PROTOBUF_LITE_LIBRARY ${PROTOBUF_LITE_LIBRARIES})
set(PROTOBUF_LITE_LIBRARY_DEBUG ${PROTOBUF_LITE_LIBRARY})
