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
# Always invoke `FIND_PACKAGE(Protobuf)` for importing function protobuf_generate_cpp
if(NOT WIN32)
  find_package(Protobuf QUIET)
endif()

unset_var(PROTOBUF_INCLUDE_DIR)
unset_var(PROTOBUF_FOUND)
unset_var(PROTOBUF_PROTOC_EXECUTABLE)
unset_var(PROTOBUF_PROTOC_LIBRARY)
unset_var(PROTOBUF_LITE_LIBRARY)
unset_var(PROTOBUF_LIBRARY)
unset_var(PROTOBUF_INCLUDE_DIR)
unset_var(Protobuf_PROTOC_EXECUTABLE)
function(protobuf_generate_python SRCS)
  # shameless copy from https://github.com/Kitware/CMake/blob/master/Modules/FindProtobuf.cmake
  if(NOT ARGN)
    message(
      SEND_ERROR
        "Error: PROTOBUF_GENERATE_PYTHON() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
    set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
  endif()

  if(DEFINED Protobuf_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    if(NOT PROTOBUF_GENERATE_CPP_APPEND_PATH)
      get_filename_component(FIL_DIR ${FIL} DIRECTORY)
      if(FIL_DIR)
        set(FIL_WE "${FIL_DIR}/${FIL_WE}")
      endif()
    endif()
    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py")
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --python_out
              ${CMAKE_CURRENT_BINARY_DIR} ${_protobuf_include_path} ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${PROTOBUF_PROTOC_EXECUTABLE}
      COMMENT "Running Python protocol buffer compiler on ${FIL}"
      VERBATIM)
  endforeach()

  set(${SRCS}
      ${${SRCS}}
      PARENT_SCOPE)
endfunction()

# Print and set the protobuf library information,
# finish this cmake process and exit from this file.
macro(PROMPT_PROTOBUF_LIB)
  set(protobuf_DEPS ${ARGN})

  message(STATUS "Protobuf protoc executable: ${PROTOBUF_PROTOC_EXECUTABLE}")
  message(STATUS "Protobuf-lite library: ${PROTOBUF_LITE_LIBRARY}")
  message(STATUS "Protobuf library: ${PROTOBUF_LIBRARY}")
  message(STATUS "Protoc library: ${PROTOBUF_PROTOC_LIBRARY}")
  message(STATUS "Protobuf version: ${PROTOBUF_VERSION}")
  include_directories(${PROTOBUF_INCLUDE_DIR})

  # Assuming that all the protobuf libraries are of the same type.
  if(${PROTOBUF_LIBRARY} MATCHES ${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(protobuf_LIBTYPE STATIC)
  elseif(${PROTOBUF_LIBRARY} MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$")
    set(protobuf_LIBTYPE SHARED)
  else()
    message(FATAL_ERROR "Unknown library type: ${PROTOBUF_LIBRARY}")
  endif()

  add_library(protobuf ${protobuf_LIBTYPE} IMPORTED GLOBAL)
  set_property(TARGET protobuf PROPERTY IMPORTED_LOCATION ${PROTOBUF_LIBRARY})

  add_library(protobuf_lite ${protobuf_LIBTYPE} IMPORTED GLOBAL)
  set_property(TARGET protobuf_lite PROPERTY IMPORTED_LOCATION
                                             ${PROTOBUF_LITE_LIBRARY})

  add_library(libprotoc ${protobuf_LIBTYPE} IMPORTED GLOBAL)
  set_property(TARGET libprotoc PROPERTY IMPORTED_LOCATION ${PROTOC_LIBRARY})

  add_executable(protoc IMPORTED GLOBAL)
  set_property(TARGET protoc PROPERTY IMPORTED_LOCATION
                                      ${PROTOBUF_PROTOC_EXECUTABLE})
  # FIND_Protobuf.cmake uses `Protobuf_PROTOC_EXECUTABLE`.
  # make `protobuf_generate_cpp` happy.
  set(Protobuf_PROTOC_EXECUTABLE ${PROTOBUF_PROTOC_EXECUTABLE})

  foreach(dep ${protobuf_DEPS})
    add_dependencies(protobuf ${dep})
    add_dependencies(protobuf_lite ${dep})
    add_dependencies(libprotoc ${dep})
    add_dependencies(protoc ${dep})
  endforeach()

  return()
endmacro()
macro(SET_PROTOBUF_VERSION)
  exec_program(
    ${PROTOBUF_PROTOC_EXECUTABLE} ARGS
    --version
    OUTPUT_VARIABLE PROTOBUF_VERSION)
  string(REGEX MATCH "[0-9]+.[0-9]+" PROTOBUF_VERSION "${PROTOBUF_VERSION}")
endmacro()

set(PROTOBUF_ROOT
    ""
    CACHE PATH "Folder contains protobuf")
if(WIN32)
  set(PROTOBUF_ROOT ${THIRD_PARTY_PATH}/install/protobuf)
endif()

if(NOT "${PROTOBUF_ROOT}" STREQUAL "")
  find_path(
    PROTOBUF_INCLUDE_DIR google/protobuf/message.h
    PATHS ${PROTOBUF_ROOT}/include
    NO_DEFAULT_PATH)
  find_library(
    PROTOBUF_LIBRARY protobuf libprotobuf.lib
    PATHS ${PROTOBUF_ROOT}/lib
    NO_DEFAULT_PATH)
  find_library(
    PROTOBUF_LITE_LIBRARY protobuf-lite libprotobuf-lite.lib
    PATHS ${PROTOBUF_ROOT}/lib
    NO_DEFAULT_PATH)
  find_library(
    PROTOBUF_PROTOC_LIBRARY protoc libprotoc.lib
    PATHS ${PROTOBUF_ROOT}/lib
    NO_DEFAULT_PATH)
  find_program(
    PROTOBUF_PROTOC_EXECUTABLE protoc
    PATHS ${PROTOBUF_ROOT}/bin
    NO_DEFAULT_PATH)
  if(PROTOBUF_INCLUDE_DIR
     AND PROTOBUF_LIBRARY
     AND PROTOBUF_LITE_LIBRARY
     AND PROTOBUF_PROTOC_LIBRARY
     AND PROTOBUF_PROTOC_EXECUTABLE)
    set(PROTOBUF_FOUND true)
    message(STATUS "Using custom protobuf library in ${PROTOBUF_ROOT}.")
    set_protobuf_version()
    prompt_protobuf_lib()
  endif()
endif()

function(build_protobuf TARGET_NAME BUILD_FOR_HOST)
  string(REPLACE "extern_" "" TARGET_DIR_NAME "${TARGET_NAME}")
  set(PROTOBUF_PREFIX_DIR ${THIRD_PARTY_PATH}/${TARGET_DIR_NAME})
  set(PROTOBUF_SOURCE_DIR
      ${THIRD_PARTY_PATH}/${TARGET_DIR_NAME}/src/${TARGET_NAME})
  set(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${TARGET_DIR_NAME})

  set(${TARGET_NAME}_INCLUDE_DIR
      "${PROTOBUF_INSTALL_DIR}/include"
      PARENT_SCOPE)
  set(PROTOBUF_INCLUDE_DIR
      "${PROTOBUF_INSTALL_DIR}/include"
      PARENT_SCOPE)
  set(${TARGET_NAME}_LITE_LIBRARY
      "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite${CMAKE_STATIC_LIBRARY_SUFFIX}"
      PARENT_SCOPE)
  set(${TARGET_NAME}_LIBRARY
      "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}"
      PARENT_SCOPE)
  set(${TARGET_NAME}_PROTOC_LIBRARY
      "${PROTOBUF_INSTALL_DIR}/lib/libprotoc${CMAKE_STATIC_LIBRARY_SUFFIX}"
      PARENT_SCOPE)
  set(${TARGET_NAME}_PROTOC_EXECUTABLE
      "${PROTOBUF_INSTALL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}"
      PARENT_SCOPE)

  set(OPTIONAL_CACHE_ARGS "")
  set(OPTIONAL_ARGS "")
  if(BUILD_FOR_HOST)
    set(OPTIONAL_ARGS "-Dprotobuf_WITH_ZLIB=OFF")
  else()
    set(OPTIONAL_ARGS
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
        "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
        "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}"
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
        "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
        "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
        "-Dprotobuf_WITH_ZLIB=ON"
        "-DZLIB_ROOT:FILEPATH=${ZLIB_ROOT}"
        ${EXTERNAL_OPTIONAL_ARGS})
    set(OPTIONAL_CACHE_ARGS "-DZLIB_ROOT:STRING=${ZLIB_ROOT}")
  endif()
  if(WIN32)
    set(OPTIONAL_ARGS
        ${OPTIONAL_ARGS} "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
        "-DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-Dprotobuf_MSVC_STATIC_RUNTIME=${MSVC_STATIC_CRT}")
  endif()

  if(WITH_ASCEND AND NOT WITH_ASCEND_CXX11)
    set(PROTOBUF_REPOSITORY https://gitee.com/tianjianhe/protobuf.git)
    set(PROTOBUF_TAG v3.8.0)
  elseif(WITH_ASCEND_CL AND NOT WITH_ASCEND_CXX11)
    set(PROTOBUF_REPOSITORY https://gitee.com/tianjianhe/protobuf.git)
    set(PROTOBUF_TAG v3.8.0)
  elseif(WITH_IPU)
    set(PROTOBUF_REPOSITORY ${GIT_URL}/protocolbuffers/protobuf.git)
    set(PROTOBUF_TAG d750fbf648256c7c631f51ffdbf67d7c18b0114e)
  elseif(WIN32)
    set(PROTOBUF_REPOSITORY ${GIT_URL}/protocolbuffers/protobuf.git)
    # Change the tag to support building with vs2019
    set(PROTOBUF_TAG 01a05a53f40ca2ac5f0af10c6cc0810bee39b792)
  else()
    set(PROTOBUF_REPOSITORY ${GIT_URL}/protocolbuffers/protobuf.git)
    set(PROTOBUF_TAG 9f75c5aa851cd877fb0d93ccc31b8567a6706546)
    if(WITH_GPU)
      if(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0
         AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 12.0)
        set(PROTOBUF_TAG 2dc747c574b68a808ea4699d26942c8132fe2b09)
      endif()
    endif()
  endif()
  if(WITH_ARM_BRPC)
    set(ARM_PROTOBUF_URL
        "https://paddlerec.bj.bcebos.com/online_infer/arm_brpc_ubuntu18/arm_protobuf.tar.gz"
        CACHE STRING "" FORCE)
    file(
      WRITE ${PROTOBUF_SOURCE_DIR}/CMakeLists.txt
      "PROJECT(ARM_PROTOBUF)\n"
      "cmake_minimum_required(VERSION 3.0)\n"
      "install(DIRECTORY arm_protobuf/bin  arm_protobuf/include arm_protobuf/lib \n"
      "        DESTINATION . USE_SOURCE_PERMISSIONS)\n")
    ExternalProject_Add(
      ${TARGET_NAME}
      ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
      PREFIX ${PROTOBUF_PREFIX_DIR}
      DOWNLOAD_DIR ${PROTOBUF_SOURCE_DIR}
      DOWNLOAD_COMMAND rm -rf arm_protobuf.tar.gz && wget --no-check-certificate
                       ${ARM_PROTOBUF_URL} && tar zxvf arm_protobuf.tar.gz
      UPDATE_COMMAND ""
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${PROTOBUF_INSTALL_DIR}
                 -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
      CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${PROTOBUF_INSTALL_DIR}
                       -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
      BUILD_BYPRODUCTS
        ${PROTOBUF_INSTALL_DIR}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}
      BUILD_BYPRODUCTS
        ${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite${CMAKE_STATIC_LIBRARY_SUFFIX}
      BUILD_BYPRODUCTS
        ${PROTOBUF_INSTALL_DIR}/lib/libprotoc${CMAKE_STATIC_LIBRARY_SUFFIX}
      BUILD_BYPRODUCTS
        ${PROTOBUF_INSTALL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX})
  else()
    ExternalProject_Add(
      ${TARGET_NAME}
      ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
      GIT_REPOSITORY ${PROTOBUF_REPOSITORY}
      GIT_TAG ${PROTOBUF_TAG}
      PREFIX ${PROTOBUF_PREFIX_DIR}
      UPDATE_COMMAND ""
      DEPENDS zlib
      CONFIGURE_COMMAND
        ${CMAKE_COMMAND} ${PROTOBUF_SOURCE_DIR}/cmake ${OPTIONAL_ARGS}
        -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_SKIP_RPATH=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF
      CMAKE_CACHE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${PROTOBUF_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        ${OPTIONAL_CACHE_ARGS}
      BUILD_BYPRODUCTS
        ${PROTOBUF_INSTALL_DIR}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}
      BUILD_BYPRODUCTS
        ${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite${CMAKE_STATIC_LIBRARY_SUFFIX}
      BUILD_BYPRODUCTS
        ${PROTOBUF_INSTALL_DIR}/lib/libprotoc${CMAKE_STATIC_LIBRARY_SUFFIX}
      BUILD_BYPRODUCTS
        ${PROTOBUF_INSTALL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX})
  endif()
endfunction()

if(WITH_ASCEND OR WITH_ASCEND_CL)
  set(PROTOBUF_VERSION 3.8.0)
elseif(WITH_IPU)
  set(PROTOBUF_VERSION 3.6.1)
elseif(WITH_ARM_BRPC)
  set(PROTOBUF_VERSION 3.7.1-baidu-ee-common)
else()
  set(PROTOBUF_VERSION 3.1.0)
  if(WITH_GPU)
    if(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0
       AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 12.0)
      set(PROTOBUF_VERSION 3.16.0)
    endif()
  endif()
endif()

if(NOT PROTOBUF_FOUND)
  build_protobuf(extern_protobuf FALSE)

  set(PROTOBUF_INCLUDE_DIR
      ${extern_protobuf_INCLUDE_DIR}
      CACHE PATH "protobuf include directory." FORCE)
  set(PROTOBUF_LITE_LIBRARY
      ${extern_protobuf_LITE_LIBRARY}
      CACHE FILEPATH "protobuf lite library." FORCE)
  set(PROTOBUF_LIBRARY
      ${extern_protobuf_LIBRARY}
      CACHE FILEPATH "protobuf library." FORCE)
  set(PROTOBUF_PROTOC_LIBRARY
      ${extern_protobuf_PROTOC_LIBRARY}
      CACHE FILEPATH "protoc library." FORCE)

  set(PROTOBUF_PROTOC_EXECUTABLE
      ${extern_protobuf_PROTOC_EXECUTABLE}
      CACHE FILEPATH "protobuf executable." FORCE)
  # `EXTERN_PROTOBUF_DEPEND` used in cmake function `proto_library` to ensure
  # `protoc.exe` existed before calling it.
  set(EXTERN_PROTOBUF_DEPEND extern_protobuf)
  prompt_protobuf_lib(extern_protobuf)
endif()
