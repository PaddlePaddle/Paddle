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

set(PROTOBUF_VERSION 3.1)
FIND_PACKAGE(Protobuf ${PROTOBUF_VERSION})

IF(PROTOBUF_FOUND)
    EXEC_PROGRAM(${PROTOBUF_PROTOC_EXECUTABLE} ARGS --version OUTPUT_VARIABLE PROTOBUF_VERSION)
    STRING(REGEX MATCH "[0-9]+.[0-9]+" PROTOBUF_VERSION "${PROTOBUF_VERSION}")
    IF (${PROTOBUF_VERSION} VERSION_LESS "3.1.0")
        SET(PROTOBUF_FOUND OFF)
    ENDIF()
ENDIF(PROTOBUF_FOUND)

IF(NOT PROTOBUF_FOUND)
    SET(PROTOBUF_SOURCES_DIR ${THIRD_PARTY_PATH}/protobuf)
    SET(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/protobuf)
    SET(PROTOBUF_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" CACHE PATH "protobuf include directory." FORCE)

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
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
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

    LIST(APPEND external_project_dependencies protobuf)
ENDIF(NOT PROTOBUF_FOUND)

INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
