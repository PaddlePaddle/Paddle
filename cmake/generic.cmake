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
#
# To build a go static library using Golang, use the go_ prefixed version:
#
#   go_library(example STATIC)
#
# To build a go shared library using Golang, use the go_ prefixed version:
#
#   go_library(example SHARED)
#

# including binary directory for generated headers.
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(NOT APPLE AND NOT ANDROID)
    find_package(Threads REQUIRED)
    link_libraries(${CMAKE_THREAD_LIBS_INIT})
    set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -pthread -ldl -lrt")
endif(NOT APPLE AND NOT ANDROID)

function(merge_static_libs TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)

  # Get all propagation dependencies from the merged libraries
  foreach(lib ${libs})
    list(APPEND libs_deps ${${lib}_LIB_DEPENDS})
  endforeach()
  list(REMOVE_DUPLICATES libs_deps)

  # To produce a library we need at least one source file.
  # It is created by add_custom_command below and will helps
  # also help to track dependencies.
  set(target_SRCS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)

  if(APPLE) # Use OSX's libtool to merge archives
    # Make the generated dummy source file depended on all static input
    # libs. If input lib changes,the source file is touched
    # which causes the desired effect (relink).
    add_custom_command(OUTPUT ${target_SRCS}
      COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS}
      DEPENDS ${libs})

    # Generate dummy staic lib
    file(WRITE ${target_SRCS} "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
    add_library(${TARGET_NAME} STATIC ${target_SRCS})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    foreach(lib ${libs})
      # Get the file names of the libraries to be merged
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND rm "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a"
      COMMAND /usr/bin/libtool -static -o "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a" ${libfiles}
      )
  else() # general UNIX: use "ar" to extract objects and re-add to a common lib
    set(target_DIR ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.dir)

    foreach(lib ${libs})
      set(objlistfile ${target_DIR}/${lib}.objlist) # list of objects in the input library
      set(objdir ${target_DIR}/${lib}.objdir)

      add_custom_command(OUTPUT ${objdir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${objdir}
        DEPENDS ${lib})

      add_custom_command(OUTPUT ${objlistfile}
        COMMAND ${CMAKE_AR} -x "$<TARGET_FILE:${lib}>"
        COMMAND ${CMAKE_AR} -t "$<TARGET_FILE:${lib}>" > ${objlistfile}
        DEPENDS ${lib} ${objdir}
        WORKING_DIRECTORY ${objdir})

      list(APPEND target_OBJS "${objlistfile}")
    endforeach()

    # Make the generated dummy source file depended on all static input
    # libs. If input lib changes,the source file is touched
    # which causes the desired effect (relink).
    add_custom_command(OUTPUT ${target_SRCS}
      COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS}
      DEPENDS ${libs} ${target_OBJS})

    # Generate dummy staic lib
    file(WRITE ${target_SRCS} "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
    add_library(${TARGET_NAME} STATIC ${target_SRCS})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    # Get the file name of the generated library
    set(target_LIBNAME "$<TARGET_FILE:${TARGET_NAME}>")

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_AR} crs ${target_LIBNAME} `find ${target_DIR} -name '*.o'`
        COMMAND ${CMAKE_RANLIB} ${target_LIBNAME}
        WORKING_DIRECTORY ${target_DIR})
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
      # Don't need link libwarpctc.so
      if ("${cc_library_DEPS};" MATCHES "warpctc;")
        list(REMOVE_ITEM cc_library_DEPS warpctc)
        add_dependencies(${TARGET_NAME} warpctc)
      endif()
      add_dependencies(${TARGET_NAME} ${cc_library_DEPS})
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    endif()
    
    # cpplint code style
    foreach(source_file ${cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cc_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()
    add_style_check_target(${TARGET_NAME} ${cc_library_SRCS} ${cc_library_HEADERS})

  else(cc_library_SRCS)
    if(cc_library_DEPS)
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
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cc_test_SRCS})
    # Support linking flags: --whole-archive (Linux) / -force_load (MacOS)
    target_circle_link_libraries(${TARGET_NAME} ${cc_test_DEPS} paddle_gtest_main paddle_memory gtest gflags)
    if("${cc_test_DEPS}" MATCHES "ARCHIVE_START")
      list(REMOVE_ITEM cc_test_DEPS ARCHIVE_START ARCHIVE_END)
    endif()
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} paddle_gtest_main paddle_memory gtest gflags)
    add_test(NAME ${TARGET_NAME}
             COMMAND ${TARGET_NAME} ${cc_test_ARGS}
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
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
        target_link_libraries(${TARGET_NAME} ${nv_library_DEPS})
      endif()
      # cpplint code style
      foreach(source_file ${nv_library_SRCS})
        string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
          list(APPEND nv_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        endif()
      endforeach()
      add_style_check_target(${TARGET_NAME} ${nv_library_SRCS} ${nv_library_HEADERS})
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
    target_link_libraries(${TARGET_NAME} ${nv_test_DEPS} paddle_gtest_main paddle_memory gtest gflags)
    add_dependencies(${TARGET_NAME} ${nv_test_DEPS} paddle_gtest_main paddle_memory gtest gflags)
    add_test(${TARGET_NAME} ${TARGET_NAME})
  endif()
endfunction(nv_test)

function(go_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs DEPS)
  cmake_parse_arguments(go_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (go_library_SHARED OR go_library_shared)
    set(BUILD_MODE "-buildmode=c-shared")
    set(${TARGET_NAME}_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}" CACHE STRING "output library name for target ${TARGET_NAME}")
  else()
    set(BUILD_MODE "-buildmode=c-archive")
    set(${TARGET_NAME}_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE STRING "output library name for target ${TARGET_NAME}")
  endif()

  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)

  # This custom command will always run since it depends on a not
  # existing file.
  add_custom_command(
    OUTPUT dummy_rebulid_${TARGET_NAME}
    COMMAND cmake -E touch ${dummyfile}
    )
  # Create a custom target that depends on the custom command output
  # file, so the custom command can be referenced as a dependency by
  # `add_dependencies`.
  add_custom_target(rebuild_${TARGET_NAME}
    DEPENDS dummy_rebulid_${TARGET_NAME}
    )

  # Add dummy code to support `make target_name` under Terminal Command
  file(WRITE ${dummyfile} "const char *dummy_${TARGET_NAME} = \"${dummyfile}\";")
  if (go_library_SHARED OR go_library_shared)
    add_library(${TARGET_NAME} SHARED ${dummyfile})
  else()
    add_library(${TARGET_NAME} STATIC ${dummyfile})
  endif()
  if(go_library_DEPS)
    add_dependencies(${TARGET_NAME} ${go_library_DEPS})
  endif(go_library_DEPS)

  # The "source file" of the library is `${dummyfile}` which never
  # change, so the target will never rebuild. Make the target depends
  # on the custom command that touches the library "source file", so
  # rebuild will always happen.
  add_dependencies(${TARGET_NAME} rebuild_${TARGET_NAME})

  set(${TARGET_NAME}_LIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/${${TARGET_NAME}_LIB_NAME}" CACHE STRING "output library path for target ${TARGET_NAME}")

  file(GLOB GO_SOURCE RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.go")
  string(REPLACE "${PADDLE_GO_PATH}/" "" CMAKE_CURRENT_SOURCE_REL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND rm "${${TARGET_NAME}_LIB_PATH}"
    # Golang build source code
    COMMAND GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} build ${BUILD_MODE}
    -o "${${TARGET_NAME}_LIB_PATH}"
    "./${CMAKE_CURRENT_SOURCE_REL_DIR}/${GO_SOURCE}"
    # must run under GOPATH
    WORKING_DIRECTORY "${PADDLE_IN_GOPATH}/go")
  add_dependencies(${TARGET_NAME} go_vendor)
endfunction(go_library)

function(go_binary TARGET_NAME)
  set(options OPTIONAL)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(go_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  string(REPLACE "${PADDLE_GO_PATH}/" "" CMAKE_CURRENT_SOURCE_REL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_custom_command(OUTPUT ${TARGET_NAME}_timestamp
    COMMAND env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} build
    -o "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}"
    "./${CMAKE_CURRENT_SOURCE_REL_DIR}/${go_binary_SRCS}"
    WORKING_DIRECTORY "${PADDLE_IN_GOPATH}/go")
  add_custom_target(${TARGET_NAME} ALL DEPENDS go_vendor ${TARGET_NAME}_timestamp ${go_binary_DEPS})
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} DESTINATION bin)
endfunction(go_binary)

function(go_test TARGET_NAME)
  set(options OPTIONAL)
  set(oneValueArgs "")
  set(multiValueArgs DEPS)
  cmake_parse_arguments(go_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  string(REPLACE "${PADDLE_GO_PATH}" "" CMAKE_CURRENT_SOURCE_REL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  add_custom_target(${TARGET_NAME} ALL DEPENDS go_vendor ${go_test_DEPS})
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND env GOPATH=${GOPATH} ${CMAKE_Go_COMPILER} test -race
    -c -o "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}"
    ".${CMAKE_CURRENT_SOURCE_REL_DIR}"
    WORKING_DIRECTORY "${PADDLE_IN_GOPATH}/go")
  add_test(NAME ${TARGET_NAME}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endfunction(go_test)

# Modification of standard 'protobuf_generate_cpp()' with protobuf-lite support
# Usage:
#   paddle_protobuf_generate_cpp(<proto_srcs> <proto_hdrs> <proto_files>)

function(paddle_protobuf_generate_cpp SRCS HDRS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: paddle_protobuf_generate_cpp() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})

  if (MOBILE_INFERENCE)
      set(EXTRA_FLAG "lite:")  
  else()
      set(EXTRA_FLAG "") 
  endif()

  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    
    set(_protobuf_protoc_src "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    set(_protobuf_protoc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")
    list(APPEND ${SRCS} "${_protobuf_protoc_src}")
    list(APPEND ${HDRS} "${_protobuf_protoc_hdr}")
    
    add_custom_command(
      OUTPUT "${_protobuf_protoc_src}"
             "${_protobuf_protoc_hdr}"

      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} 
      -I${CMAKE_CURRENT_SOURCE_DIR}
      --cpp_out "${EXTRA_FLAG}${CMAKE_CURRENT_BINARY_DIR}" ${ABS_FIL}
      DEPENDS ${ABS_FIL} protoc
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()


function(proto_library TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(proto_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(proto_srcs)
  set(proto_hdrs)
  paddle_protobuf_generate_cpp(proto_srcs proto_hdrs ${proto_library_SRCS})
  cc_library(${TARGET_NAME} SRCS ${proto_srcs} DEPS ${proto_library_DEPS} protobuf)
endfunction()

function(py_proto_compile TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS)
  cmake_parse_arguments(py_proto_compile "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(py_srcs)
  protobuf_generate_python(py_srcs ${py_proto_compile_SRCS})
  add_custom_target(${TARGET_NAME} ALL DEPENDS ${py_srcs})
endfunction()

function(py_test TARGET_NAME)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    cmake_parse_arguments(py_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_test(NAME ${TARGET_NAME}
             COMMAND env PYTHONPATH=${PADDLE_PYTHON_BUILD_DIR}/lib-python ${py_test_ENVS}
             ${PYTHON_EXECUTABLE} -u ${py_test_SRCS} ${py_test_ARGS}
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
endfunction()

# grpc_library generate grpc code using grpc_cpp_plugin and protoc
# then build the generated protobuf code and grpc code with your
# implementation source codes together. Use SRCS argument for your
# implementation source files and PROTO argument for your .proto
# files.
#
# Usage: grpc_library(my_target SRCS my_client.cc PROTO my_target.proto DEPS my_dep)

function(grpc_library TARGET_NAME)
  set(oneValueArgs PROTO)
  set(multiValueArgs SRCS DEPS)
  set(options "")
  cmake_parse_arguments(grpc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  message(STATUS "generating grpc ${grpc_library_PROTO}")

  get_filename_component(ABS_PROTO ${grpc_library_PROTO} ABSOLUTE)
  get_filename_component(PROTO_WE ${grpc_library_PROTO} NAME_WE)
  get_filename_component(PROTO_PATH ${ABS_PROTO} PATH)

  protobuf_generate_cpp(grpc_proto_srcs grpc_proto_hdrs "${ABS_PROTO}")
  set(grpc_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_WE}.grpc.pb.cc")
  set(grpc_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_WE}.grpc.pb.h")
  cc_library("${TARGET_NAME}_proto" SRCS "${grpc_proto_srcs}")

  add_custom_command(
          OUTPUT "${grpc_grpc_srcs}" "${grpc_grpc_hdrs}"
          COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
          ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}" -I "${PROTO_PATH}"
          --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}" "${ABS_PROTO}"
          DEPENDS "${ABS_PROTO}" ${PROTOBUF_PROTOC_EXECUTABLE} extern_grpc)

  # FIXME(typhoonzero): grpc generated code do not generate virtual-dtor, mark it
  # as compiler warnings instead of error. Should try remove the warnings also.
  set_source_files_properties(
    ${grpc_grpc_srcs}
    PROPERTIES
    COMPILE_FLAGS  "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
  cc_library("${TARGET_NAME}_grpc" SRCS "${grpc_grpc_srcs}")

  set_source_files_properties(
    ${grpc_library_SRCS}
    PROPERTIES
    COMPILE_FLAGS  "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")
  cc_library("${TARGET_NAME}" SRCS "${grpc_library_SRCS}" DEPS "${TARGET_NAME}_grpc" "${TARGET_NAME}_proto" "${grpc_library_DEPS}")
endfunction()
