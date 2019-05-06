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

if(NOT APPLE)
  find_package(Threads REQUIRED)
  link_libraries(${CMAKE_THREAD_LIBS_INIT})
  set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -pthread -ldl -lrt")
endif(NOT APPLE)

set_property(GLOBAL PROPERTY FLUID_MODULES "")
# find all fluid modules is used for paddle fluid static library
# for building inference libs
function(find_fluid_modules TARGET_NAME)
  get_filename_component(__target_path ${TARGET_NAME} ABSOLUTE)
  string(REGEX REPLACE "^${PADDLE_SOURCE_DIR}/" "" __target_path ${__target_path})
  string(FIND "${__target_path}" "fluid" pos)
  if(pos GREATER 1)
    get_property(fluid_modules GLOBAL PROPERTY FLUID_MODULES)
    set(fluid_modules ${fluid_modules} ${TARGET_NAME})
    set_property(GLOBAL PROPERTY FLUID_MODULES "${fluid_modules}")
  endif()
endfunction(find_fluid_modules)


function(common_link TARGET_NAME)
  if (WITH_PROFILER)
    target_link_libraries(${TARGET_NAME} gperftools::profiler)
  endif()

  if (WITH_JEMALLOC)
    target_link_libraries(${TARGET_NAME} jemalloc::jemalloc)
  endif()
endfunction()


# find all third_party modules is used for paddle static library
# for reduce the dependency when building the inference libs.
set_property(GLOBAL PROPERTY FLUID_THIRD_PARTY)
function(find_fluid_thirdparties TARGET_NAME)
  get_filename_component(__target_path ${TARGET_NAME} ABSOLUTE)
  string(REGEX REPLACE "^${PADDLE_SOURCE_DIR}/" "" __target_path ${__target_path})
  string(FIND "${__target_path}" "third_party" pos)
  if(pos GREATER 1)
    get_property(fluid_ GLOBAL PROPERTY FLUID_THIRD_PARTY)
    set(fluid_third_partys ${fluid_third_partys} ${TARGET_NAME})
    set_property(GLOBAL PROPERTY FLUID_THIRD_PARTY "${fluid_third_partys}")
  endif()
endfunction(find_fluid_thirdparties)

function(merge_static_libs TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)

  # Get all propagation dependencies from the merged libraries
  foreach(lib ${libs})
    list(APPEND libs_deps ${${lib}_LIB_DEPENDS})
  endforeach()
  if(libs_deps)
    list(REMOVE_DUPLICATES libs_deps)
  endif()

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
  endif(APPLE)
  if(LINUX) # general UNIX: use "ar" to extract objects and re-add to a common lib
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
  endif(LINUX)
  if(WIN32) # windows do not support gcc/nvcc combined compiling. Use msvc lib.exe to merge libs.
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
    # msvc will put libarary in directory of "/Release/xxxlib" by default
    #       COMMAND cmake -E remove "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${TARGET_NAME}.lib"
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND cmake -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}"
      COMMAND lib /OUT:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib${TARGET_NAME}.lib ${libfiles}
      )
  endif(WIN32)
endfunction(merge_static_libs)

function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(WIN32)
      # add libxxx.lib prefix in windows
      set(${TARGET_NAME}_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE STRING "output library name for target ${TARGET_NAME}")
  endif(WIN32)
  if(cc_library_SRCS)
    if(cc_library_SHARED OR cc_library_shared) # build *.so
      add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
      find_fluid_modules(${TARGET_NAME})
    endif()

    if(cc_library_DEPS)
      # Don't need link libwarpctc.so
      if("${cc_library_DEPS};" MATCHES "warpctc;")
        list(REMOVE_ITEM cc_library_DEPS warpctc)
        add_dependencies(${TARGET_NAME} warpctc)
      endif()
      # Only deps libwbaes.so, not link
      if("${cc_library_DEPS};" MATCHES "wbaes;")
        list(REMOVE_ITEM cc_library_DEPS wbaes)
        if(NOT "${TARGET_NAME}" MATCHES "dynload_wbaes")
          list(APPEND cc_library_DEPS dynload_wbaes)
        endif()
        add_dependencies(${TARGET_NAME} wbaes)
      endif()
      # Only deps libmklml.so, not link
      if("${cc_library_DEPS};" MATCHES "mklml;")
        list(REMOVE_ITEM cc_library_DEPS mklml)
        if(NOT "${TARGET_NAME}" MATCHES "dynload_mklml")
          list(APPEND cc_library_DEPS dynload_mklml)
        endif()
        add_dependencies(${TARGET_NAME} mklml)
        if(WIN32)
          target_link_libraries(${TARGET_NAME} ${MKLML_IOMP_LIB})
        else(WIN32)
          target_link_libraries(${TARGET_NAME} "-L${MKLML_LIB_DIR} -liomp5 -Wl,--as-needed")
        endif(WIN32)
      endif()
      # remove link to python, see notes at:
      # https://github.com/pybind/pybind11/blob/master/docs/compiling.rst#building-manually
      if("${cc_library_DEPS};" MATCHES "python;")
        list(REMOVE_ITEM cc_library_DEPS python)
        add_dependencies(${TARGET_NAME} python)
        if(WIN32)
          target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})
        else()
          target_link_libraries(${TARGET_NAME} "-Wl,-undefined,dynamic_lookup")
        endif(WIN32)
      endif()
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
      add_dependencies(${TARGET_NAME} ${cc_library_DEPS})
      common_link(${TARGET_NAME})
    endif()

    # cpplint code style
    foreach(source_file ${cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cc_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()
  else(cc_library_SRCS)
    if(cc_library_DEPS)
      merge_static_libs(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL_ERROR "Please specify source files or libraries in cc_library(${TARGET_NAME} ...).")
    endif()
  endif(cc_library_SRCS)
endfunction(cc_library)

# The link operation under windows may exceeds the maximum characters limit, simply break the link command
# into multiple link opeartion can fix that, say
# original:
#     lib /out:target.lib a.lib b.lib c.lib d.lib
# after:
#    1. lib /out:dummy_lib_1.lib a.lib b.lib
#    2. lib /out:dummy_lib_2.lib c.lib d.lib
#    1. lib /out:target.lib dummy_lib_1.lib dummy_lib_2.lib
function(sep_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(sep_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(dummy_index 1)
  set(dummy_offset 1)
  # the dummy target would be consisted of limit size libraries
  set(dummy_limit 50)
  list(LENGTH sep_library_DEPS sep_all_len)
  foreach(v ${sep_library_DEPS})
    list(APPEND dummy_list ${v})
    list(LENGTH dummy_list listlen )
    if ((${listlen} GREATER ${dummy_limit}) OR (${dummy_offset} EQUAL ${sep_all_len}))
      message("create dummy library ${TARGET_NAME}_dummy_lib_${dummy_index} for ${TARGET_NAME}")
      cc_library(${TARGET_NAME}_dummy_lib_${dummy_index} STATIC DEPS ${dummy_list})
      foreach(i ${dummy_list})
        list(REMOVE_AT dummy_list 0)
      endforeach()
      list(APPEND ${TARGET_NAME}_dummy_list ${TARGET_NAME}_dummy_lib_${dummy_index})
      MATH(EXPR dummy_index "${dummy_index}+1")
    endif()
    MATH(EXPR dummy_offset "${dummy_offset}+1")
  endforeach()
  if(${sep_library_SHARED})
    cc_library(${TARGET_NAME} SHARED SRCS ${sep_library_SRCS} DEPS ${${TARGET_NAME}_dummy_list})
  else(${sep_library_SHARED})
    cc_library(${TARGET_NAME} STATIC SRCS ${sep_library_SRCS} DEPS ${${TARGET_NAME}_dummy_list})
  endif(${sep_library_SHARED})
endfunction(sep_library)

function(cc_binary TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_executable(${TARGET_NAME} ${cc_binary_SRCS})
  if(cc_binary_DEPS)
    target_link_libraries(${TARGET_NAME} ${cc_binary_DEPS})
    add_dependencies(${TARGET_NAME} ${cc_binary_DEPS})
    common_link(${TARGET_NAME})
  endif()
  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(${TARGET_NAME} ${os_dependency_modules})
endfunction(cc_binary)

function(cc_test TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cc_test_SRCS})
    if(WIN32)
      if("${cc_test_DEPS};" MATCHES "python;")
        list(REMOVE_ITEM cc_test_DEPS python)
        target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})
      endif()
    endif(WIN32)
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} ${os_dependency_modules} paddle_gtest_main lod_tensor memory gtest gflags glog)
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} paddle_gtest_main lod_tensor memory gtest gflags glog)
    common_link(${TARGET_NAME})
    add_test(NAME ${TARGET_NAME}
             COMMAND ${TARGET_NAME} ${cc_test_ARGS}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if (${cc_test_SERIAL})
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    set_property(TEST ${TARGET_NAME} PROPERTY ENVIRONMENT FLAGS_cpu_deterministic=true)
    set_property(TEST ${TARGET_NAME} PROPERTY ENVIRONMENT FLAGS_init_allocated_mem=true)
    set_property(TEST ${TARGET_NAME} PROPERTY ENVIRONMENT FLAGS_limit_of_tmp_allocation=4294967296) # 4G
    set_property(TEST ${TARGET_NAME} PROPERTY ENVIRONMENT FLAGS_cudnn_deterministic=true)
    # No unit test should exceed 10 minutes.
    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 600)
  endif()
endfunction(cc_test)

# cc_test without default dependencies
function(raw_cc_test TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cc_test_SRCS})
    if(WIN32)
      if("${cc_test_DEPS};" MATCHES "python;")
        list(REMOVE_ITEM cc_test_DEPS python)
        target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})
      endif()
    endif(WIN32)
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} ${os_dependency_modules} lite_gtest_main gtest gflags glog)
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} lite_gtest_main gtest gflags glog)
    common_link(${TARGET_NAME})
    add_test(NAME ${TARGET_NAME}
            COMMAND ${TARGET_NAME} ${cc_test_ARGS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if (${cc_test_SERIAL})
      set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    # No unit test should exceed 10 minutes.
    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 600)
  endif()
endfunction(raw_cc_test)

function(lite_cc_test args)
  if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    message(STATUS "building lite raw test: ${args}")
    raw_cc_test(${args} ${ARGN})
  else()
    message(STATUS "building lite heavy test: ${args}")
    cc_test(${args} ${ARGN})
  endif()
endfunction()

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
        find_fluid_modules(${TARGET_NAME})
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
      common_link(${TARGET_NAME})
    endif()
  endif()
endfunction(nv_binary)

function(nv_test TARGET_NAME)
  if (WITH_GPU AND WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(nv_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cuda_add_executable(${TARGET_NAME} ${nv_test_SRCS})
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(${TARGET_NAME} ${nv_test_DEPS} paddle_gtest_main lod_tensor memory gtest gflags glog ${os_dependency_modules})
    add_dependencies(${TARGET_NAME} ${nv_test_DEPS} paddle_gtest_main lod_tensor memory gtest gflags glog)
    common_link(${TARGET_NAME})
    add_test(${TARGET_NAME} ${TARGET_NAME})
    if (nv_test_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    set_property(TEST ${TARGET_NAME} PROPERTY ENVIRONMENT FLAGS_cpu_deterministic=true)
    set_property(TEST ${TARGET_NAME} PROPERTY ENVIRONMENT FLAGS_init_allocated_mem=true)
    set_property(TEST ${TARGET_NAME} PROPERTY ENVIRONMENT FLAGS_limit_of_tmp_allocation=4294967296) # 4G
    set_property(TEST ${TARGET_NAME} PROPERTY ENVIRONMENT FLAGS_cudnn_deterministic=true)
  endif()
endfunction(nv_test)

function(hip_library TARGET_NAME)
  if (WITH_AMD_GPU)
    set(options STATIC static SHARED shared)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(hip_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(_sources ${hip_library_SRCS})
    HIP_PREPARE_TARGET_COMMANDS(${TARGET_NAME} OBJ _generated_files _source_files ${_sources} HIPCC_OPTIONS ${_hipcc_options} HCC_OPTIONS ${_hcc_options} NVCC_OPTIONS ${_nvcc_options})
    if(_source_files)
      list(REMOVE_ITEM _sources ${_source_files})
    endif()
    if(hip_library_SRCS)
      if (hip_library_SHARED OR hip_library_shared) # build *.so
        add_library(${TARGET_NAME} SHARED ${_cmake_options} ${_generated_files} ${_sources})
        set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE HIP)
      else()
        add_library(${TARGET_NAME} STATIC ${_cmake_options} ${_generated_files} ${_sources})
        set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(${TARGET_NAME} /opt/rocm/hip/lib/libhip_hcc.so /opt/rocm/hip/lib/libhip_device.a /opt/rocm/rccl/lib/librccl.so /opt/rocm/hiprand/lib/libhiprand.so)
        find_fluid_modules(${TARGET_NAME})
      endif()
      if("${hip_library_DEPS}" MATCHES "ARCHIVE_START")
        # Support linking flags: --whole-archive (Linux) / -force_load (MacOS).
        # WARNING: Please don't use ARCHIVE_START&ARCHIVE_END if TARGET_NAME will be linked by other libraries.
        target_circle_link_libraries(${TARGET_NAME} ${hip_library_DEPS})
        list(REMOVE_ITEM hip_library_DEPS ARCHIVE_START ARCHIVE_END)
      else()
        target_link_libraries(${TARGET_NAME} ${hip_library_DEPS})
      endif()
      # cpplint code style
      foreach(source_file ${hip_library_SRCS})
        string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
          list(APPEND hip_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        endif()
      endforeach()
    else(hip_library_SRCS)
      if (hip_library_DEPS)
        merge_static_libs(${TARGET_NAME} ${hip_library_DEPS})
      else()
        message(FATAL "Please specify source file or library in nv_library.")
      endif()
    endif(hip_library_SRCS)
  endif()
endfunction(hip_library)

function(hip_binary TARGET_NAME)
  if (WITH_AMD_GPU)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(hip_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    hip_add_executable(${TARGET_NAME} ${hip_binary_SRCS})
    if(hip_binary_DEPS)
      target_link_libraries(${TARGET_NAME} ${hip_binary_DEPS})
      add_dependencies(${TARGET_NAME} ${hip_binary_DEPS})
      common_link(${TARGET_NAME})
    endif()
  endif()
endfunction(hip_binary)

function(hip_test TARGET_NAME)
  if (WITH_AMD_GPU AND WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(hip_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(_sources ${hip_test_SRCS})
    HIP_PREPARE_TARGET_COMMANDS(${TARGET_NAME} OBJ _generated_files _source_files ${_sources} HIPCC_OPTIONS ${_hipcc_options} HCC_OPTIONS ${_hcc_options} NVCC_OPTIONS ${_nvcc_options})
    if(_source_files)
      list(REMOVE_ITEM _sources ${_source_files})
    endif()
    add_executable(${TARGET_NAME} ${_cmake_options} ${_generated_files} ${_sources})
    set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE HIP)
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(${TARGET_NAME} ${hip_test_DEPS} paddle_gtest_main memory gtest gflags ${os_dependency_modules})
    add_dependencies(${TARGET_NAME} ${hip_test_DEPS} paddle_gtest_main memory gtest gflags)
    common_link(${TARGET_NAME})
    add_test(${TARGET_NAME} ${TARGET_NAME})
  endif()
endfunction(hip_test)

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
    common_link(${TARGET_NAME})
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
      --cpp_out "${CMAKE_CURRENT_BINARY_DIR}" ${ABS_FIL}
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
             COMMAND ${CMAKE_COMMAND} -E env FLAGS_init_allocated_mem=true FLAGS_cudnn_deterministic=true
             FLAGS_cpu_deterministic=true FLAGS_limit_of_tmp_allocation=4294967296  # 4G
             PYTHONPATH=${PADDLE_BINARY_DIR}/python ${py_test_ENVS}
             ${PYTHON_EXECUTABLE} -u ${py_test_SRCS} ${py_test_ARGS}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # No unit test should exceed 10 minutes.
    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 600)
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

  #FIXME(putcn): the follwoing line is supposed to generate *.pb.h and cc, but
  # somehow it didn't. line 602 to 604 is to patching this. Leaving this here
  # for now to enable dist CI.
  paddle_protobuf_generate_cpp(grpc_proto_srcs grpc_proto_hdrs "${ABS_PROTO}")
  set(grpc_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_WE}.grpc.pb.cc")
  set(grpc_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_WE}.grpc.pb.h")
  cc_library("${TARGET_NAME}_proto" SRCS "${grpc_proto_srcs}")

  add_custom_command(
          OUTPUT "${grpc_grpc_srcs}" "${grpc_grpc_hdrs}"
          COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
          ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}" -I "${PROTO_PATH}"
          --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}" "${ABS_PROTO}"
          COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
          ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}" -I "${PROTO_PATH}"
          "${ABS_PROTO}"
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


function(brpc_library TARGET_NAME)
  set(oneValueArgs PROTO)
  set(multiValueArgs SRCS DEPS)
  set(options "")
  cmake_parse_arguments(brpc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  message(STATUS "generating brpc ${brpc_library_PROTO}")

  get_filename_component(ABS_PROTO ${brpc_library_PROTO} ABSOLUTE)
  get_filename_component(PROTO_WE ${brpc_library_PROTO} NAME_WE)
  get_filename_component(PROTO_PATH ${ABS_PROTO} PATH)

  paddle_protobuf_generate_cpp(brpc_proto_srcs brpc_proto_hdrs "${ABS_PROTO}")
  cc_library("${TARGET_NAME}_proto" SRCS "${brpc_proto_srcs}")
  cc_library("${TARGET_NAME}" SRCS "${brpc_library_SRCS}" DEPS "${TARGET_NAME}_proto" "${brpc_library_DEPS}")
endfunction()
