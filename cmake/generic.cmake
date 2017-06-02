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
#
# To simplify the build process of PaddlePaddle, we defined couple of
# fundamental abstractions, e.g., how to build library, binary and
# test in C++, CUDA and Go.
#
# -------------------------------------------
#     C++        CUDA C++       Go
# -------------------------------------------
# cc_library    nv_library   go_library
# cc_binary     nv_binary    go_binary
# cc_test       nv_test      go_test
# -------------------------------------------
#
# cmake_parse_arguments can help us to achieve this goal.
# https://cmake.org/cmake/help/v3.0/module/CMakeParseArguments.html
#
# cc_library|nv_library(<target_name>
# [STATIC SHARED OBJECT]
#   SRCS <file>...
#   OBJS <objs>...
#   DEPS <libs>...)
#
# cc_library and nv_library can generate *.o, *.a, or *.so
# if the corresponding keyword OBJECT, STATIC or SHARED is specified.
#
# cc_binary|nv_binary(<target_name>
#   SRCS <file>...
#   OBJS <objs>...
#   DEPS <libs>...)
#
# cc_binary and nv_binary can build souce code and link the dependent
# libraries to generate a binary.
#
# cc_test|nv_test(<target_name>
#   SRCS <file>...
#   OBJS <objs>...
#   DEPS <libs>...)
#
# cc_test and nv_test can build test code, link gtest and other dependent
# libraries to generate test suite.
#
# For example, in one folder, it contains
#   ddim{.h, .cc, _test.cc, _test.cu}
#   place{.h, cc, _test.cc}
#
# We can add build script as follows: 
# 
# cc_library(place OBJECT
#    SRCS place.cc)
#
# place.cc -> place.o
# cc_library's OBJECT OPTION will generate place.o.
#
# cc_test(place_test
#    SRCS place_test.cc
#    OBJS place
#    DEPS glog gflags)
#
# place_test.cc, place.o, glog, gflags -> place_test
# cc_test will combine place_test.cc, place.o with libglog.a
# and libgflags.a to generate place_test.
#
# cc_library(ddim OBJECT
#    SRCS ddim.cc)
#
# ddim.cc -> ddim.o
# cc_library's OBJECT OPTION will generate ddim.o.
#
# cc_test(ddim_test
#    SRCS ddim_test.cc
#    OBJS ddim)
#
# ddim_test.cc, ddim.o -> ddim_test
# cc_test will build ddim_test.cc with ddim.o to generate ddim_test.
#
# nv_test(dim_test
#    SRCS dim_test.cu
#    OBJS ddim)
#
# dim_test.cu, ddim.o -> dim_test
# nv_test will build dim_test.cu with ddim.o to generate dim_test.
#
# cc_library(majel
#    OBJS place ddim)
#
# place.o, ddim.o -> libmajel.a
# cc_library's default OPTION is STATIC. It will archive place.o
# and ddim.o to generate libmajel.a.
#

if(NOT APPLE)
    find_package(Threads REQUIRED)
    link_libraries(${CMAKE_THREAD_LIBS_INIT})
endif(NOT APPLE)

function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared OBJECT object)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS OBJS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(__objs "")
  foreach(object ${cc_library_OBJS})
    set(__objs $<TARGET_OBJECTS:${object}> ${__objs})
  endforeach()
  if (cc_library_SHARED OR cc_library_shared) # build *.so
    add_library(${TARGET_NAME} SHARED ${cc_library_SRCS} ${__objs})
  else()
    if (cc_library_OBJECT OR cc_library_object) # build *.o
      add_library(${TARGET_NAME} OBJECT ${cc_library_SRCS} ${__objs})
    else() # default build *.a
      add_library(${TARGET_NAME} ${cc_library_SRCS} ${__objs})
    endif()
  endif()
  if (cc_library_DEPS OR cc_library_OBJS)
    add_dependencies(${TARGET_NAME} ${cc_library_DEPS} ${cc_library_OBJS})
  endif()
endfunction(cc_library)

function(cc_binary TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS OBJS)
  cmake_parse_arguments(cc_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(__objs "")
  foreach(object ${cc_binary_OBJS})
    list(APPEND __objs $<TARGET_OBJECTS:${object}>)
  endforeach()
  add_executable(${TARGET_NAME} ${cc_binary_SRCS} ${__objs})
  if(cc_binary_DEPS)
    target_link_libraries(${TARGET_NAME} ${cc_binary_DEPS})
  endif()
  if(cc_binary_DEPS OR cc_binary_OBJS)
    add_dependencies(${TARGET_NAME} ${cc_binary_DEPS} ${cc_binary_OBJS})
  endif()
endfunction(cc_binary)

function(cc_test TARGET_NAME)
  if (WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS OBJS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(__objs "")
    foreach(object ${cc_test_OBJS})
      list(APPEND __objs $<TARGET_OBJECTS:${object}>)
    endforeach()
    add_executable(${TARGET_NAME} ${cc_test_SRCS} ${__objs})
    target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} gtest gtest_main)
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} ${cc_test_OBJS} gtest gtest_main)
    add_test(${TARGET_NAME} ${TARGET_NAME})
  endif(WITH_TESTING)
endfunction(cc_test)

function(nv_library TARGET_NAME)
  if (WITH_GPU)
    set(options STATIC static SHARED shared OBJECT object)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS OBJS)
    cmake_parse_arguments(nv_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(__objs "")
    foreach(object ${nv_library_OBJS})
      set(__objs $<TARGET_OBJECTS:${object}> ${__objs})
    endforeach()
    if (nv_library_SHARED OR nv_library_shared) # build *.so
      cuda_add_library(${TARGET_NAME} SHARED ${nv_library_SRCS} ${__objs})
    else()
      if (cc_library_OBJECT OR cc_library_object) # build *.o
        cuda_compile(${TARGET_NAME} ${nv_library_SRCS} ${__objs})
      else() # default build *.a
        cuda_add_library(${TARGET_NAME} STATIC ${nv_library_SRCS} ${__objs})
      endif()
    endif()
    if (nv_library_DEPS OR nv_library_OBJS)
      add_dependencies(${TARGET_NAME} ${nv_library_DEPS} ${nv_library_OBJS})
    endif()
  endif(WITH_GPU)
endfunction(nv_library)

function(nv_binary TARGET_NAME)
  if (WITH_GPU)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS OBJS)
    cmake_parse_arguments(nv_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(__objs "")
    foreach(object ${nv_binary_OBJS})
      set(__objs $<TARGET_OBJECTS:${object}> ${__objs})
    endforeach()
    cuda_add_executable(${TARGET_NAME} ${nv_binary_SRCS} ${__objs})
    if(nv_binary_DEPS)
      target_link_libraries(${TARGET_NAME} ${nv_binary_DEPS})
    endif()
    if(nv_binary_DEPS OR nv_binary_OBJS)
      add_dependencies(${TARGET_NAME} ${nv_binary_DEPS} ${nv_binary_OBJS})
    endif()
  endif(WITH_GPU)
endfunction(nv_binary)

function(nv_test TARGET_NAME)
  if (WITH_GPU AND WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS OBJS)
    cmake_parse_arguments(nv_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(__objs "")
    foreach(object ${nv_test_OBJS})
      set(__objs $<TARGET_OBJECTS:${object}> ${__objs})
    endforeach()    
    cuda_add_executable(${TARGET_NAME} ${nv_test_SRCS} ${__objs})
    target_link_libraries(${TARGET_NAME} ${nv_test_DEPS} gtest gtest_main)
    add_dependencies(${TARGET_NAME} ${nv_test_DEPS} ${nv_test_OBJS} gtest gtest_main)
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
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
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

# go_extern will download extern go project.
# go_extern(target_name extern_source)
# go_extern(go_redis github.com/hoisie/redis)
function(go_extern TARGET_NAME)
  add_custom_target(${TARGET_NAME} env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} get ${ARGN})
endfunction(go_extern)
