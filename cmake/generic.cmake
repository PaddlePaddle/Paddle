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
  add_dependencies(${TARGET_NAME} ${cc_binary_DEPS} ${external_project_dependencies})
  target_link_libraries(${TARGET_NAME} ${cc_binary_DEPS})
endfunction(cc_binary)

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
  if (${cc_library_OPTIONAL} STREQUAL "STATIC")
    add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
  else()
    add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
  endif()
  add_dependencies(${TARGET_NAME} ${cc_library_DEPS} ${external_project_dependencies})
endfunction(cc_library)

# The dependency to target tensor implies that if any of
# tensor{.h,.cc,_test.cc} is changed, tensor_test need to be re-built.
# cc_test(tensor_test
#   SRCS
#   tensor_test.cc
#   DEPS
#   tensor)
function(cc_test TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_executable(${TARGET_NAME} ${cc_test_SRCS})
  add_dependencies(${TARGET_NAME} ${cc_test_DEPS} ${external_project_dependencies})
  target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} ${GTEST_MAIN_LIBRARIES} ${GTEST_LIBRARIES})
  add_test(${TARGET_NAME} ${TARGET_NAME})
endfunction(cc_test)
