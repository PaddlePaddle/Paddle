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


# generic.cmake defines CMakes functions that look like Bazel's
# building rules (https://bazel.build/).
#
# 
# -------------------------------------------
#     C++        CUDA C++       Go
# -------------------------------------------
# cc_library    nv_library   go_library
# cc_binary     nv_binary    go_binary
# cc_test       nv_test      go_test
# -------------------------------------------
# 
# To build a static library example.a from example.cc using the system
#  compiler (like GCC):
# 
#   cc_library(example SRCS example.cc)
# 
# To build a static library example.a from multiple source files
# example{1,2,3}.cc:
# 
#   cc_library(example SRCS example1.cc example2.cc example3.cc)
# 
# To build a shared library example.so from example.cc:
# 
#   cc_library(example SHARED SRCS example.cc)
# 
# To build a library using Nvidia's NVCC from .cu file(s), use the nv_
# prefixed version:
# 
#   nv_library(example SRCS example.cu)
# 
# To specify that a library new_example.a depends on other libraies:
# 
#   cc_library(new_example SRCS new_example.cc DEPS example)
# 
# Static libraries can be composed of other static libraries:
# 
#   cc_library(composed DEPS dependent1 dependent2 dependent3)
# 
# To build an executable binary file from some source files and
# dependent libraries:
# 
#   cc_binary(example SRCS main.cc something.cc DEPS example1 example2)
# 
# To build an executable binary file using NVCC, use the nv_ prefixed
# version:
# 
#   nv_binary(example SRCS main.cc something.cu DEPS example1 example2)
# 
# To build a unit test binary, which is an executable binary with
# GoogleTest linked:
# 
#   cc_test(example_test SRCS example_test.cc DEPS example)
# 
# To build a unit test binary using NVCC, use the nv_ prefixed version:
# 
#   nv_test(example_test SRCS example_test.cu DEPS example)
#
# It is pretty often that executable and test binaries depend on
# pre-defined external libaries like glog and gflags defined in
# /cmake/external/*.cmake:
#
#   cc_test(example_test SRCS example_test.cc DEPS example glog gflags)

if(WITH_GPU)
  add_definitions(-DPADDLE_WITH_GPU)
endif()

if(NOT APPLE)
    find_package(Threads REQUIRED)
    link_libraries(${CMAKE_THREAD_LIBS_INIT})
endif(NOT APPLE)

function(merge_static_libs TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)

  # First get the file names of the libraries to be merged
  foreach(lib ${libs})
    get_target_property(libtype ${lib} TYPE)
    if(NOT libtype STREQUAL "STATIC_LIBRARY")
      message(FATAL_ERROR "merge_static_libs can only process static libraries")
    endif()
    set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
  endforeach()

  if(APPLE) # Use OSX's libtool to merge archives
    add_custom_target(${TARGET_NAME}_archive
      COMMAND libtool -static -o "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a" ${libfiles}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${libs}
      )
    add_library(${TARGET_NAME} STATIC IMPORTED GLOBAL)
    set_property(TARGET ${TARGET_NAME} PROPERTY
      IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a")
    add_dependencies(${TARGET_NAME} ${TARGET_NAME}_archive)
	else() # general UNIX: use "ar" to extract objects and re-add to a common lib
    foreach(lib ${libs})
      set(objlistfile ${lib}.objlist) # list of objects in the input library
      set(objdir ${lib}.objdir)

      add_custom_command(OUTPUT ${objdir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${objdir})

      add_custom_command(OUTPUT ${objlistfile}
        COMMAND ${CMAKE_AR} -x "$<TARGET_FILE:${lib}>"
        COMMAND ${CMAKE_AR} -t "$<TARGET_FILE:${lib}>" > ../${objlistfile}
        DEPENDS ${lib} ${objdir}
        WORKING_DIRECTORY ${objdir})

      # Empty dummy source file that goes into merged library
      set(mergebase ${lib}.mergebase.c)
      add_custom_command(OUTPUT ${mergebase}
        COMMAND ${CMAKE_COMMAND} -E touch ${mergebase}
        DEPENDS ${objlistfile})

      list(APPEND mergebases "${mergebase}")
    endforeach()

    # We need a target for the output merged library
    add_library(${TARGET_NAME} STATIC ${mergebases})
    set(outlibfile "$<TARGET_FILE:${TARGET_NAME}>")

    foreach(lib ${libs})
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_AR} ru ${outlibfile} @"../${objlistfile}"
      WORKING_DIRECTORY ${objdir})
    endforeach()

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_RANLIB} ${outlibfile})
  endif()
endfunction(merge_static_libs)

function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if (cc_library_SRCS)
    if (cc_library_SHARED OR cc_library_shared) # build *.so
      add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
    endif()
    if (cc_library_DEPS)
      add_dependencies(${TARGET_NAME} ${cc_library_DEPS})
    endif()
  else(cc_library_SRCS)
    if (cc_library_DEPS)
      merge_static_libs(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL "Please specify source file or library in cc_library.")
    endif()
  endif(cc_library_SRCS)
endfunction(cc_library)

function(cc_binary TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_executable(${TARGET_NAME} ${cc_binary_SRCS})
  if(cc_binary_DEPS)
    target_link_libraries(${TARGET_NAME} ${cc_binary_DEPS})
    add_dependencies(${TARGET_NAME} ${cc_binary_DEPS})
  endif()
endfunction(cc_binary)

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

function(nv_library TARGET_NAME)
  if (WITH_GPU)
    set(options STATIC static SHARED shared)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(nv_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(nv_library_SRCS)
      if (nv_library_SHARED OR nv_library_shared) # build *.so
        cuda_add_library(${TARGET_NAME} SHARED ${nv_library_SRCS})
      else()
          cuda_add_library(${TARGET_NAME} STATIC ${nv_library_SRCS})
      endif()
      if (nv_library_DEPS)
        add_dependencies(${TARGET_NAME} ${nv_library_DEPS})
      endif()
    else(nv_library_SRCS)
      if (nv_library_DEPS)
        merge_static_libs(${TARGET_NAME} ${nv_library_DEPS})
      else()
        message(FATAL "Please specify source file or library in nv_library.")
      endif()
    endif(nv_library_SRCS)
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
