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
# Always invoke `FIND_PACKAGE(Protobuf)` for importing function protobuf_generate_cpp
FIND_PACKAGE(Protobuf QUIET)
SET(PROTOBUF_FOUND "OFF")


# Print and set the protobuf library information,
# finish this cmake process and exit from this file.
macro(PROMPT_PROTOBUF_LIB)
    SET(protobuf_DEPS ${ARGN})

    MESSAGE(STATUS "Protobuf protoc executable: ${PROTOBUF_PROTOC_EXECUTABLE}")
    MESSAGE(STATUS "Protobuf library: ${PROTOBUF_LIBRARY}")
    MESSAGE(STATUS "Protobuf version: ${PROTOBUF_VERSION}")
    INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

    # Assuming that all the protobuf libraries are of the same type.
    IF(${PROTOBUF_LIBRARY} MATCHES "${CMAKE_STATIC_LIBRARY_SUFFIX}$")
        SET(protobuf_LIBTYPE STATIC)
    ELSEIF(${PROTOBUF_LIBRARY} MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$")
        SET(protobuf_LIBTYPE SHARED)
    ELSE()
        MESSAGE(FATAL_ERROR "Unknown library type: ${PROTOBUF_LIBRARY}")
    ENDIF()

    ADD_LIBRARY(protobuf ${protobuf_LIBTYPE} IMPORTED GLOBAL)
    SET_PROPERTY(TARGET protobuf PROPERTY IMPORTED_LOCATION ${PROTOBUF_LIBRARY})

    ADD_LIBRARY(protobuf_lite ${protobuf_LIBTYPE} IMPORTED GLOBAL)
    SET_PROPERTY(TARGET protobuf_lite PROPERTY IMPORTED_LOCATION ${PROTOBUF_LITE_LIBRARY})

    ADD_LIBRARY(libprotoc ${protobuf_LIBTYPE} IMPORTED GLOBAL)
    SET_PROPERTY(TARGET libprotoc PROPERTY IMPORTED_LOCATION ${PROTOC_LIBRARY})

    ADD_EXECUTABLE(protoc IMPORTED GLOBAL)
    SET_PROPERTY(TARGET protoc PROPERTY IMPORTED_LOCATION ${PROTOBUF_PROTOC_EXECUTABLE})
    # FIND_Protobuf.cmake uses `Protobuf_PROTOC_EXECUTABLE`.
    # make `protobuf_generate_cpp` happy.
    SET(Protobuf_PROTOC_EXECUTABLE ${PROTOBUF_PROTOC_EXECUTABLE})

    FOREACH(dep ${protobuf_DEPS})
        ADD_DEPENDENCIES(protobuf ${dep})
        ADD_DEPENDENCIES(protobuf_lite ${dep})
        ADD_DEPENDENCIES(libprotoc ${dep})
        ADD_DEPENDENCIES(protoc ${dep})
    ENDFOREACH()

    LIST(APPEND external_project_dependencies protobuf)
    RETURN()
endmacro()
macro(SET_PROTOBUF_VERSION)
    EXEC_PROGRAM(${PROTOBUF_PROTOC_EXECUTABLE} ARGS --version OUTPUT_VARIABLE PROTOBUF_VERSION)
    STRING(REGEX MATCH "[0-9]+.[0-9]+" PROTOBUF_VERSION "${PROTOBUF_VERSION}")
endmacro()

set(PROTOBUF_ROOT "" CACHE PATH "Folder contains protobuf")
if (NOT "${PROTOBUF_ROOT}" STREQUAL "")
    find_path(PROTOBUF_INCLUDE_DIR google/protobuf/message.h PATHS ${PROTOBUF_ROOT}/include)
    find_library(PROTOBUF_LIBRARY protobuf PATHS ${PROTOBUF_ROOT}/lib)
    find_library(PROTOBUF_LITE_LIBRARY protobuf-lite PATHS ${PROTOBUF_ROOT}/lib)
    find_library(PROTOBUF_PROTOC_LIBRARY protoc PATHS ${PROTOBUF_ROOT}/lib)
    find_program(PROTOBUF_PROTOC_EXECUTABLE protoc PATHS ${PROTOBUF_ROOT}/bin)
    if (PROTOBUF_INCLUDE_DIR AND PROTOBUF_LIBRARY AND PROTOBUF_LITE_LIBRARY AND PROTOBUF_PROTOC_LIBRARY AND PROTOBUF_PROTOC_EXECUTABLE)
        message(STATUS "Using custom protobuf library in ${PROTOBUF_ROOT}.")
        SET_PROTOBUF_VERSION()
        PROMPT_PROTOBUF_LIB()
    else()
        message(WARNING "Cannot find protobuf library in ${PROTOBUF_ROOT}.")
    endif()
endif()

FUNCTION(build_protobuf TARGET_NAME BUILD_FOR_HOST)
    STRING(REPLACE "extern_" "" TARGET_DIR_NAME "${TARGET_NAME}")
    SET(PROTOBUF_SOURCES_DIR ${THIRD_PARTY_PATH}/${TARGET_DIR_NAME})
    SET(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${TARGET_DIR_NAME})

    SET(${TARGET_NAME}_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(PROTOBUF_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(${TARGET_NAME}_LITE_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite${CMAKE_STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotoc${CMAKE_STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_EXECUTABLE
        "${PROTOBUF_INSTALL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}"
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
ENDFUNCTION()

SET(PROTOBUF_VERSION 3.1)
IF(NOT CMAKE_CROSSCOMPILING)
    FIND_PACKAGE(Protobuf ${PROTOBUF_VERSION})

    IF(PROTOBUF_FOUND)
        SET_PROTOBUF_VERSION()
        IF("${PROTOBUF_VERSION}" VERSION_LESS "3.1.0")
            SET(PROTOBUF_FOUND OFF)
        ELSE()
            PROMPT_PROTOBUF_LIB()
        ENDIF()
    ENDIF(PROTOBUF_FOUND)
ELSE()
    build_protobuf(protobuf_host TRUE)
    LIST(APPEND external_project_dependencies protobuf_host)

    SET(PROTOBUF_PROTOC_EXECUTABLE ${protobuf_host_PROTOC_EXECUTABLE}
        CACHE FILEPATH "protobuf executable." FORCE)
ENDIF()

IF(NOT PROTOBUF_FOUND)
    build_protobuf(extern_protobuf FALSE)

    SET(PROTOBUF_INCLUDE_DIR ${extern_protobuf_INCLUDE_DIR}
        CACHE PATH "protobuf include directory." FORCE)
    SET(PROTOBUF_LITE_LIBRARY ${extern_protobuf_LITE_LIBRARY}
        CACHE FILEPATH "protobuf lite library." FORCE)
    SET(PROTOBUF_LIBRARY ${extern_protobuf_LIBRARY}
        CACHE FILEPATH "protobuf library." FORCE)
    SET(PROTOBUF_PROTOC_LIBRARY ${extern_protobuf_PROTOC_LIBRARY}
        CACHE FILEPATH "protoc library." FORCE)

    IF(CMAKE_CROSSCOMPILING)
        PROMPT_PROTOBUF_LIB(protobuf_host extern_protobuf)
    ELSE()
        SET(PROTOBUF_PROTOC_EXECUTABLE ${extern_protobuf_PROTOC_EXECUTABLE}
            CACHE FILEPATH "protobuf executable." FORCE)
        PROMPT_PROTOBUF_LIB(extern_protobuf)
    ENDIF()
ENDIF(NOT PROTOBUF_FOUND)
