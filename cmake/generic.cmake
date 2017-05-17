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
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  add_custom_target(${TARGET_NAME}_lib ALL DEPENDS ${TARGET_NAME}_timestamp ${go_library_DEPS})
  add_library(${TARGET_NAME} STATIC IMPORTED)
  set_target_properties(${TARGET_NAME} PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME})
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
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})  
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
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})  
  add_custom_target(${TARGET_NAME} ALL DEPENDS ${TARGET_NAME}_timestamp ${go_test_DEPS})  
  add_test(${TARGET_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
endfunction(go_test)
