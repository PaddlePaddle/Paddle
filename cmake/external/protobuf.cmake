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

FUNCTION(build_protobuf TARGET_NAME BUILD_FOR_HOST)
    SET(PROTOBUF_SOURCES_DIR ${THIRD_PARTY_PATH}/${TARGET_NAME})
    SET(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${TARGET_NAME})

    SET(${TARGET_NAME}_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(PROTOBUF_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(${TARGET_NAME}_LITE_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite${STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf${STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotoc${STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_EXECUTABLE
        "${PROTOBUF_INSTALL_DIR}/bin/protoc${EXECUTABLE_SUFFIX}"
         PARENT_SCOPE)

    SET(OPTIONAL_CACHE_ARGS "")
    SET(OPTIONAL_ARGS "")
    IF(BUILD_FOR_HOST)
        SET(OPTIONAL_ARGS "-Dprotobuf_WITH_ZLIB=OFF")
    ELSE()
        SET(OPTIONAL_ARGS
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
            "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
            "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
            "-Dprotobuf_WITH_ZLIB=ON"
            "-DZLIB_ROOT:FILEPATH=${ZLIB_ROOT}")
        SET(OPTIONAL_CACHE_ARGS "-DZLIB_ROOT:STRING=${ZLIB_ROOT}")
    ENDIF()

    ExternalProject_Add(
        ${TARGET_NAME}
        ${EXTERNAL_PROJECT_LOG_ARGS}
        PREFIX          ${PROTOBUF_SOURCES_DIR}
        UPDATE_COMMAND  ""
        DEPENDS         zlib
        GIT_REPOSITORY  "https://github.com/google/protobuf.git"
        GIT_TAG         "9f75c5aa851cd877fb0d93ccc31b8567a6706546"
        CONFIGURE_COMMAND
        ${CMAKE_COMMAND} ${PROTOBUF_SOURCES_DIR}/src/${TARGET_NAME}/cmake
            ${OPTIONAL_ARGS}
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
            ${OPTIONAL_CACHE_ARGS}
    )

    LIST(APPEND external_project_dependencies ${TARGET_NAME} PARENT_SCOPE)
ENDFUNCTION()

SET(PROTOBUF_VERSION 3.1)
IF(NOT CMAKE_CROSSCOMPILING)
    FIND_PACKAGE(Protobuf ${PROTOBUF_VERSION})

    IF(PROTOBUF_FOUND)
        EXEC_PROGRAM(${PROTOBUF_PROTOC_EXECUTABLE} ARGS --version OUTPUT_VARIABLE PROTOBUF_VERSION)
        STRING(REGEX MATCH "[0-9]+.[0-9]+" PROTOBUF_VERSION "${PROTOBUF_VERSION}")
        IF("${PROTOBUF_VERSION}" VERSION_LESS "3.1.0")
            SET(PROTOBUF_FOUND OFF)
        ENDIF()
    ENDIF(PROTOBUF_FOUND)
ELSE()
    build_protobuf(protobuf_host TRUE)
    SET(PROTOBUF_PROTOC_EXECUTABLE ${protobuf_host_PROTOC_EXECUTABLE}
        CACHE FILEPATH "protobuf executable." FORCE)
ENDIF()

IF(NOT PROTOBUF_FOUND)
    build_protobuf(protobuf FALSE)
    SET(PROTOBUF_INCLUDE_DIR ${protobuf_INCLUDE_DIR}
        CACHE PATH "protobuf include directory." FORCE)
    IF(NOT CMAKE_CROSSCOMPILING)
        SET(PROTOBUF_PROTOC_EXECUTABLE ${protobuf_PROTOC_EXECUTABLE}
            CACHE FILEPATH "protobuf executable." FORCE)
    ENDIF()
    SET(PROTOBUF_LITE_LIBRARY ${protobuf_LITE_LIBRARY} CACHE FILEPATH "protobuf lite library." FORCE)
    SET(PROTOBUF_LIBRARY ${protobuf_LIBRARY} CACHE FILEPATH "protobuf library." FORCE)
    SET(PROTOBUF_PROTOC_LIBRARY ${protobuf_PROTOC_LIBRARY} CACHE FILEPATH "protoc library." FORCE)
ENDIF(NOT PROTOBUF_FOUND)

MESSAGE(STATUS "Protobuf protoc executable: ${PROTOBUF_PROTOC_EXECUTABLE}")
MESSAGE(STATUS "Protobuf library: ${PROTOBUF_LIBRARY}")
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
