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


# To simplify the build process of PaddlePaddle, we defined couple of
# fundamental abstractions, e.g., how to build library, binary and
# test in C++, CUDA and Go.
#
# -------------------------------------------
#    C++	      CUDA C++	      Go
# -------------------------------------------
# cc_library	 nv_library	  go_library
# cc_binary  	 nv_binary	  go_binary
# cc_test        nv_test	  go_test
# -------------------------------------------
#
# cmake_parse_arguments can help us to achieve this goal.
# https://cmake.org/cmake/help/v3.0/module/CMakeParseArguments.html
#

if(NOT APPLE)
    find_package(Threads REQUIRED)
    link_libraries(${CMAKE_THREAD_LIBS_INIT})
endif(NOT APPLE)

# cc_library parses tensor.cc and figures out that target also depend on tensor.h.
# cc_library(tensor
#   SRCS
#   tensor.cc
#   DEPS
#   variant)
function(cc_library TARGET_NAME)
  set(options OPTIONAL)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if (${cc_library_OPTIONAL} STREQUAL "SHARED")
    add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
  else()
    add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
  endif()
  if (cc_library_DEPS)
    add_dependencies(${TARGET_NAME} ${cc_library_DEPS})
  endif()
endfunction(cc_library)

# cc_binary parses tensor.cc and figures out that target also depend on tensor.h.
# cc_binary(tensor
#   SRCS
#   tensor.cc)
function(cc_binary TARGET_NAME)
  set(options OPTIONAL)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_executable(${TARGET_NAME} ${cc_binary_SRCS})
  if(cc_binary_DEPS)
    target_link_libraries(${TARGET_NAME} ${cc_binary_DEPS})
    add_dependencies(${TARGET_NAME} ${cc_binary_DEPS})
  endif()
endfunction(cc_binary)

# The dependency to target tensor implies that if any of
# tensor{.h,.cc,_test.cc} is changed, tensor_test need to be re-built.
# cc_test(tensor_test
#   SRCS
#   tensor_test.cc
#   DEPS
#   tensor)
function(cc_test TARGET_NAME)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cc_test_SRCS})
    target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} gtest gtest_main)
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} gtest gtest_main)
    add_test(${TARGET_NAME} ${TARGET_NAME})
  endif()
endfunction(cc_test)

# Suppose that ops.cu includes global functions that take Tensor as
# their parameters, so ops depend on tensor. This implies that if
# any of tensor.{h.cc}, ops.{h,cu} is changed, ops need to be re-built.
# nv_library(ops
#   SRCS
#   ops.cu
#   DEPS
#   tensor)
function(nv_library TARGET_NAME)
  if (WITH_GPU)
    set(options OPTIONAL)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(nv_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if (${nv_library_OPTIONAL} STREQUAL "SHARED")
      cuda_add_library(${TARGET_NAME} SHARED ${nv_library_SRCS})
    else()
      cuda_add_library(${TARGET_NAME} STATIC ${nv_library_SRCS})
    endif()
    if (nv_library_DEPS)
      add_dependencies(${TARGET_NAME} ${nv_library_DEPS})
    endif()
  endif()
endfunction(nv_library)

function(nv_binary TARGET_NAME)
  if (WITH_GPU)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(nv_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cuda_add_executable(${TARGET_NAME} ${nv_binary_SRCS})
    if(nv_binary_DEPS)
      target_link_libraries(${TARGET_NAME} ${nv_binary_DEPS})
      add_dependencies(${TARGET_NAME} ${nv_binary_DEPS})
    endif()
  endif()
endfunction(nv_binary)

# The dependency to target tensor implies that if any of
# ops{.h,.cu,_test.cu} is changed, ops_test need to be re-built.
# nv_test(ops_test
#   SRCS
#   ops_test.cu
#   DEPS
#   ops)
function(nv_test TARGET_NAME)
  if (WITH_GPU AND WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(nv_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cuda_add_executable(${TARGET_NAME} ${nv_test_SRCS})
    target_link_libraries(${TARGET_NAME} ${nv_test_DEPS} gtest gtest_main)
    add_dependencies(${TARGET_NAME} ${nv_test_DEPS} gtest gtest_main)
    add_test(${TARGET_NAME} ${TARGET_NAME})
  endif()
endfunction(nv_test)

set(GOPATH "${CMAKE_CURRENT_BINARY_DIR}/go")
file(MAKE_DIRECTORY ${GOPATH})

# Because api.go defines a GO wrapper to ops and tensor, it depends on
# both.  This implies that if any of tensor.{h,cc}, ops.{h,cu}, or
# api.go is changed, api need to be re-built.
# go_library(api
#   SRCS
#   api.go
#   DEPS
#   tensor # Because ops depend on tensor, this line is optional.
#   ops)
function(go_library TARGET_NAME)
  set(options OPTIONAL)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(go_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if (${go_library_OPTIONAL} STREQUAL "SHARED")
    set(BUILD_MODE "-buildmode=c-shared")
    if(APPLE)
      set(LIB_NAME "lib${TARGET_NAME}.dylib")
    else()
      set(LIB_NAME "lib${TARGET_NAME}.so")
    endif()
  else()
    set(BUILD_MODE "-buildmode=c-archive")
    set(LIB_NAME "lib${TARGET_NAME}.a")
  endif()
  add_custom_command(OUTPUT ${TARGET_NAME}_timestamp
    COMMAND env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} build ${BUILD_MODE}
    -o "${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}"
    ${go_library_SRCS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_custom_target(${TARGET_NAME}_lib ALL DEPENDS ${TARGET_NAME}_timestamp ${go_library_DEPS})
  add_library(${TARGET_NAME} STATIC IMPORTED)
  set_property(TARGET ${TARGET_NAME} PROPERTY
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}")
  add_dependencies(${TARGET_NAME} ${TARGET_NAME}_lib)
endfunction(go_library)

function(go_binary TARGET_NAME)
  set(options OPTIONAL)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(go_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_custom_command(OUTPUT ${TARGET_NAME}_timestamp
    COMMAND env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} build
    -o "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}"
    ${go_library_SRCS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_custom_target(${TARGET_NAME} ALL DEPENDS ${TARGET_NAME}_timestamp ${go_binary_DEPS})
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} DESTINATION bin)
endfunction(go_binary)

function(go_test TARGET_NAME)
  set(options OPTIONAL)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(go_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_custom_command(OUTPUT ${TARGET_NAME}_timestamp
    COMMAND env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} test
    -c -o "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}"
    ${go_test_SRCS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_custom_target(${TARGET_NAME} ALL DEPENDS ${TARGET_NAME}_timestamp ${go_test_DEPS})
  add_test(${TARGET_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
endfunction(go_test)

# go_extern will download extern go project.
# go_extern(target_name extern_source)
# go_extern(go_redis github.com/hoisie/redis)
function(go_extern TARGET_NAME)
  add_custom_target(${TARGET_NAME} env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} get ${ARGN})
endfunction(go_extern)
