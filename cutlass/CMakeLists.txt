# Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)

if(cutlass_LOADED)
  # If CUTLASS has been previously fetched and loaded, don't do it again.
  return()
else()
  set(cutlass_LOADED ON)
  set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "CUTLASS Repository Directory")
endif()

message(STATUS "CMake Version: ${CMAKE_VERSION}")

project(CUTLASS VERSION 2.9.0 LANGUAGES CXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/CUDA.cmake)

if (CUDA_VERSION VERSION_LESS 10.2)
  message(WARNING "CUTLASS ${CUTLASS_VERSION} requires CUDA 10.2 or higher, and strongly recommends CUDA 11.0 or higher.")
elseif (CUDA_VERSION VERSION_LESS 11.0)
  message(WARNING "CUTLASS ${CUTLASS_VERSION} support for CUDA ${CUDA_VERSION} is deprecated, please use CUDA 11.0 or higher.")
endif()

find_package(Doxygen QUIET)

#
# CUTLASS 2.x requires C++11
#
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CUTLASS_NATIVE_CUDA)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --std=c++11)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX install CACHE PATH "Default installation location." FORCE)
endif()

message(STATUS "Default Install Location: ${CMAKE_INSTALL_PREFIX}")

set(CUTLASS_ENABLE_HEADERS_ONLY OFF CACHE BOOL "Enable only the header library")

if(CUTLASS_ENABLE_HEADERS_ONLY)
  set(CUTLASS_ENABLE_EXAMPLES_INIT OFF)
  set(CUTLASS_ENABLE_TOOLS_INIT ON)
  set(CUTLASS_ENABLE_LIBRARY_INIT OFF)
else()
  set(CUTLASS_ENABLE_EXAMPLES_INIT ON)
  set(CUTLASS_ENABLE_TOOLS_INIT ON)
  set(CUTLASS_ENABLE_LIBRARY_INIT ON)
endif()

set(CUTLASS_TEST_UNIT_ENABLE_WARNINGS OFF CACHE BOOL "Enable warnings on waived unit tests.")

set(CUTLASS_ENABLE_EXAMPLES ${CUTLASS_ENABLE_EXAMPLES_INIT} CACHE BOOL "Enable CUTLASS Examples")
set(CUTLASS_ENABLE_TOOLS ${CUTLASS_ENABLE_TOOLS_INIT} CACHE BOOL "Enable CUTLASS Tools")
set(CUTLASS_ENABLE_LIBRARY ${CUTLASS_ENABLE_LIBRARY_INIT} CACHE BOOL "Enable CUTLASS Library")
set(CUTLASS_ENABLE_PROFILER ${CUTLASS_ENABLE_LIBRARY} CACHE BOOL "Enable CUTLASS Profiler")

if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  set(CUTLASS_ENABLE_TESTS_INIT ${CUTLASS_ENABLE_LIBRARY}})
else()
  set(CUTLASS_ENABLE_TESTS_INIT OFF)
endif()

set(CUTLASS_ENABLE_TESTS ${CUTLASS_ENABLE_TESTS_INIT} CACHE BOOL "Enable CUTLASS Tests")

if (CUTLASS_ENABLE_TESTS)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake)
endif()

set(CUTLASS_NVCC_ARCHS_SUPPORTED "")
if (NOT CUDA_VERSION VERSION_LESS 7.5)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 53)
endif()
if (NOT CUDA_VERSION VERSION_LESS 8.0)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 60 61)
endif()
if (NOT CUDA_VERSION VERSION_LESS 9.0)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 70)
endif()
if (NOT CUDA_VERSION VERSION_LESS 9.2)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 72)
endif()
if (NOT CUDA_VERSION VERSION_LESS 10.0)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 75)
endif()
if (NOT CUDA_VERSION VERSION_LESS 11.0)
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 80)
endif()
if (NOT CUDA_VERSION VERSION_LESS 11.1 AND NOT CUDA_COMPILER MATCHES "[Cc]lang")
  list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 86)
endif()
set(CUTLASS_NVCC_ARCHS ${CUTLASS_NVCC_ARCHS_SUPPORTED} CACHE STRING "The SM architectures requested.")
set(CUTLASS_NVCC_ARCHS_ENABLED ${CUTLASS_NVCC_ARCHS} CACHE STRING "The SM architectures to build code for.")

# Special policy introduced in CMake 3.13
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

include(GNUInstallDirs)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

###################################################################################################
#
# Configure CMake variables
#
###################################################################################################

message(STATUS "CUDA Compilation Architectures: ${CUTLASS_NVCC_ARCHS_ENABLED}")

if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  # By default we want to build in Release mode to ensure that we're getting best performance.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (DEFINED CMAKE_DEBUG_POSTFIX)
  set(CUTLASS_LIBRARY_DEBUG_POSTFIX_INIT ${CMAKE_DEBUG_POSTFIX})
else()
  set(CUTLASS_LIBRARY_DEBUG_POSTFIX_INIT .debug)
endif()
set(CUTLASS_LIBRARY_DEBUG_POSTFIX ${CUTLASS_LIBRARY_DEBUG_POSTFIX_INIT} CACHE STRING "Default postfix value for debug libraries")

if(WIN32)
  # On Windows we link against the shared (DLL) runtime. Change gtest settings to match this.
  set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib" FORCE)
endif()

if (WIN32)
  # Enable more warnings and treat as errors
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=/W3 -Xcompiler=/WX)

  # Disable warning on Unicode characters
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=/wd4819)

  # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=/fp:strict)
endif(WIN32)

if (${CUTLASS_NVCC_VERBOSE})
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -v)
endif()

#
# CUTLASS NAMESPACE 
#
set(CUTLASS_NAMESPACE "cutlass" CACHE STRING "Top level namespace of CUTLASS")

set(CUTLASS_NVCC_EMBED_CUBIN ON CACHE BOOL "Embed compiled CUDA kernel binaries into executables.")
set(CUTLASS_NVCC_EMBED_PTX ON CACHE BOOL "Embed compiled PTX into executables.")
set(CUTLASS_NVCC_KEEP OFF CACHE BOOL "Keep intermediate files generated by NVCC.")
set(CUTLASS_ENABLE_F16C OFF CACHE BOOL "Enable F16C x86 extensions in host code.")

#
# CUTLASS generator cmake configuration
#
set(CUTLASS_LIBRARY_OPERATIONS "all" CACHE STRING "Comma delimited list of operation name filters. Default '' means all operations are enabled.")
set(CUTLASS_LIBRARY_KERNELS "" CACHE STRING "Comma delimited list of kernel name filters. If unspecified, only the largest tile size is enabled. If 'all' is specified, all kernels are enabled.")
set(CUTLASS_LIBRARY_IGNORE_KERNELS "" CACHE STRING "Comma delimited list of kernel names to exclude from build.")

# Test Levels L0, L1, L2
set(CUTLASS_TEST_LEVEL "0" CACHE STRING "Level of tests to compile.")

set(CUTLASS_TEST_ENABLE_CACHED_RESULTS ON CACHE BOOL "Enable caching and reuse of test results in unit tests")

set_property(CACHE CUTLASS_TEST_LEVEL PROPERTY STRINGS 0 1 2)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_TEST_LEVEL=${CUTLASS_TEST_LEVEL})
list(APPEND CUTLASS_CUDA_CLANG_FLAGS -DCUTLASS_TEST_LEVEL=${CUTLASS_TEST_LEVEL})

if (CUTLASS_TEST_ENABLE_CACHED_RESULTS)
  message(STATUS "Enable caching of reference results in conv unit tests")
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1)
endif()

set(CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED ON CACHE BOOL "Enable/Disable rigorous conv problem sizes in conv unit tests")

if (CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED)
  message(STATUS "Enable rigorous conv problem sizes in conv unit tests")
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1)
endif()

#
# CUDA 10.1 introduces "mma" in PTX performing collective matrix multiply operations.
#

if (CUDA_VERSION VERSION_LESS 10.1)
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT OFF)
else()
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT ON)
endif()

# Trace levels for debugging
set(CUTLASS_DEBUG_TRACE_LEVEL "0" CACHE STRING "Level of debug tracing to perform.")
list(APPEND CUTLASS_CUDA_NVCC_FLAGS -DCUTLASS_DEBUG_TRACE_LEVEL=${CUTLASS_DEBUG_TRACE_LEVEL})

set(CUTLASS_ENABLE_TENSOR_CORE_MMA ${CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT} CACHE BOOL
  "Enable PTX mma instruction for collective matrix multiply operations.")

#
# NOTE: running with asan and CUDA requires the following environment variable:
#
#  ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0
#
# without the above environment setting, an error like the following may be generated:
#
#  *** Error: Could not detect active GPU device ID [out of memory]
#  ...
#  ==9149==ERROR: LeakSanitizer: detected memory leaks
#  ...
#
if(ENABLE_ASAN)  # https://github.com/google/sanitizers/wiki/AddressSanitizer
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --compiler-options=-fsanitize=address --compiler-options=-fno-omit-frame-pointer)
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address")
endif()

###################################################################################################
#
# Configure CUDA build options
#
###################################################################################################

if(CUTLASS_NVCC_EMBED_PTX)
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS --cuda-include-ptx=all)
endif()

if (CUTLASS_ENABLE_TENSOR_CORE_MMA)
  list(APPEND CUTLASS_CUDA_FLAGS -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1)
endif()

if (NOT MSVC AND CUTLASS_NVCC_KEEP)
  # MSVC flow handles caching already, but for other generators we handle it here.
  set(CUTLASS_NVCC_KEEP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp CACHE PATH "Location to store NVCC scratch files")
  file(MAKE_DIRECTORY ${CUTLASS_NVCC_KEEP_DIR})
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --keep -v) # --keep-dir may not work with nvcc for some directories.
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -save-temps=${CUTLASS_NVCC_KEEP_DIR})
endif()

if (CUTLASS_ENABLE_F16C AND NOT CMAKE_CROSSCOMPILING)
  list(APPEND CUTLASS_CUDA_FLAGS -DCUTLASS_ENABLE_F16C=1)
  if ((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=-mf16c)
  elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC"))
    list(APPEND CUTLASS_CUDA_NVCC_FLAGS -Xcompiler=/arch:AVX2)
  endif()
endif()

list(APPEND CUTLASS_CUDA_NVCC_FLAGS $<$<BOOL:${UNIX}>:-Xcompiler=-Wconversion>)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS $<$<BOOL:${UNIX}>:-Xcompiler=-fno-strict-aliasing>)

# Don't leak lineinfo in release builds
if (NOT CMAKE_BUILD_TYPE MATCHES "Release")
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -gmlt)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS -lineinfo)
endif()

#Report CUDA build flags
if (CUDA_COMPILER MATCHES "[Cc]lang")
  if(CUTLASS_CUDA_CLANG_FLAGS)
    message(STATUS "Using CLANG flags: ${CUTLASS_CUDA_CLANG_FLAGS}")
  endif()
else()
  if(CUTLASS_CUDA_NVCC_FLAGS)
    message(STATUS "Using NVCC flags: ${CUTLASS_CUDA_NVCC_FLAGS}")
  endif()
endif()

if(CUDA_COMPILER MATCHES "[Cc]lang")
  if( NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    message(FATAL_ERROR "Clang CUDA compilation requires Clang CXX compilation. Currently CMAKE_CXX_COMPILER is ${CMAKE_CXX_COMPILER_ID}" )
  endif()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "Clang 7.0+ required for GPU compilation")
  endif()

  # There are numerous Clang versions that can work with each CUDA toolkit and the 
  # the checks are not very useful so we are turning them off and using testing to 
  # ensure the various combinations work properly.

  list(APPEND CUTLASS_CUDA_CLANG_FLAGS --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -D__NV_NO_HOST_COMPILER_CHECK=1)
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -Wno-unknown-cuda-version)

  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -mllvm -pragma-unroll-threshold=100000)
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -mllvm -unroll-threshold=5000)
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -Wno-unused-command-line-argument)

  string(REPLACE "." ";" CUDA_VERSION_PARTS ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET CUDA_VERSION_PARTS 0 CUDA_VERSION_MAJOR)
  list(GET CUDA_VERSION_PARTS 1 CUDA_VERSION_MINOR)
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -D__CUDACC_VER_MAJOR__=${CUDA_VERSION_MAJOR} -D__CUDACC_VER_MINOR__=${CUDA_VERSION_MINOR})


  # needed for libcublasLt.so in case it's installed in the same location as libcudart.so
  # dynamic linker can find it if linker sets RPATH (forced by --disable-new-tags)
  # Otherwise linker uses RUNPATH and that does not propagate to loaded libs.
  list(APPEND CUTLASS_CUDA_CLANG_FLAGS -Wl,--disable-new-dtags)

  link_libraries(nvidia::cudart)
endif()

# Support for 128-bit integers if using NVIDIA C++ compiler 
if (${CMAKE_CXX_COMPILER_ID} MATCHES "PGI" OR ${CMAKE_CXX_COMPILER_ID} MATCHES "NVHPC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Mint128 ")
endif()

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  # CMake 3.18 added support for CUDA_ARCHITECTURES target property. We will use this
  # property for CMake 3.18+, so we request the NEW behavior for correct compatibility.
  # https://cmake.org/cmake/help/v3.18/policy/CMP0104.html#policy:CMP0104 
  cmake_policy(SET CMP0104 NEW)
endif()

function(cutlass_apply_cuda_gencode_flags TARGET)

  set(NVCC_FLAGS)
  set(CLANG_FLAGS)
  set(__CMAKE_CUDA_ARCHS)
  foreach(ARCH ${CUTLASS_NVCC_ARCHS_ENABLED})
    list(APPEND CLANG_FLAGS --cuda-gpu-arch=sm_${ARCH})
    set(CODES)
    if(CUTLASS_NVCC_EMBED_CUBIN)
      list(APPEND CODES sm_${ARCH})
      list(APPEND __CMAKE_CUDA_ARCHS ${ARCH}-real)
    endif()
    if(CUTLASS_NVCC_EMBED_PTX)
      list(APPEND CODES compute_${ARCH})
      list(APPEND __CMAKE_CUDA_ARCHS ${ARCH}-virtual)
    endif()
    list(JOIN CODES "," CODES_STR)
    list(APPEND NVCC_FLAGS -gencode=arch=compute_${ARCH},code=[${CODES_STR}])
  endforeach()

  if (CUDA_COMPILER MATCHES "[Cc]lang")
    target_compile_options(
      ${TARGET}
      PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${CLANG_FLAGS}>
      )
  elseif(CMAKE_VERSION GREATER_EQUAL 3.18)
    set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES ${__CMAKE_CUDA_ARCHS})
  else()
    target_compile_options(
      ${TARGET}
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
      )
  endif()

endfunction()

# Cache the flags so they are available when the function below is called anywhere globally. 

set(__CUTLASS_CUDA_FLAGS ${CUTLASS_CUDA_FLAGS} CACHE INTERNAL "")
set(__CUTLASS_CUDA_FLAGS_RELEASE ${CUTLASS_CUDA_FLAGS_RELEASE} CACHE INTERNAL "")
set(__CUTLASS_CUDA_FLAGS_RELWITHDEBINFO ${CUTLASS_CUDA_FLAGS_RELWITHDEBINFO} CACHE INTERNAL "")
set(__CUTLASS_CUDA_FLAGS_DEBUG ${CUTLASS_CUDA_FLAGS_DEBUG} CACHE INTERNAL "")
set(__CUTLASS_CUDA_CLANG_FLAGS ${CUTLASS_CUDA_CLANG_FLAGS} CACHE INTERNAL "")
set(__CUTLASS_CUDA_CLANG_FLAGS_RELEASE ${CUTLASS_CUDA_CLANG_FLAGS_RELEASE} CACHE INTERNAL "")
set(__CUTLASS_CUDA_CLANG_FLAGS_RELWITHDEBINFO ${CUTLASS_CUDA_CLANG_FLAGS_RELWITHDEBINFO} CACHE INTERNAL "")
set(__CUTLASS_CUDA_CLANG_FLAGS_DEBUG ${CUTLASS_CUDA_CLANG_FLAGS_DEBUG} CACHE INTERNAL "")
set(__CUTLASS_CUDA_NVCC_FLAGS ${CUTLASS_CUDA_NVCC_FLAGS} CACHE INTERNAL "")
set(__CUTLASS_CUDA_NVCC_FLAGS_RELEASE ${CUTLASS_CUDA_NVCC_FLAGS_RELEASE} CACHE INTERNAL "")
set(__CUTLASS_CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUTLASS_CUDA_NVCC_FLAGS_RELWITHDEBINFO} CACHE INTERNAL "")
set(__CUTLASS_CUDA_NVCC_FLAGS_DEBUG ${CUTLASS_CUDA_NVCC_FLAGS_DEBUG} CACHE INTERNAL "")

function(cutlass_apply_standard_compile_options TARGET)

  if(CUDA_COMPILER MATCHES "[Cc]lang")
    set(CUDA_COMPILE_LANGUAGE CXX)
    set(_FLAGS ${__CUTLASS_CUDA_FLAGS} ${__CUTLASS_CUDA_CLANG_FLAGS})
    set(_FLAGS_RELEASE ${__CUTLASS_CUDA_FLAGS_RELEASE} ${__CUTLASS_CUDA_CLANG_FLAGS_RELEASE})
    set(_FLAGS_RELWITHDEBINFO ${__CUTLASS_CUDA_FLAGS_RELWITHDEBINFO} ${__CUTLASS_CUDA_CLANG_FLAGS_RELWITHDEBINFO})
    set(_FLAGS_DEBUG ${__CUTLASS_CUDA_FLAGS_DEBUG} ${__CUTLASS_CUDA_CLANG_FLAGS_DEBUG})
  else()
    set(CUDA_COMPILE_LANGUAGE CUDA)
    set(_FLAGS ${__CUTLASS_CUDA_FLAGS} ${__CUTLASS_CUDA_NVCC_FLAGS})
    set(_FLAGS_RELEASE ${__CUTLASS_CUDA_FLAGS_RELEASE} ${__CUTLASS_CUDA_NVCC_FLAGS_RELEASE})
    set(_FLAGS_RELWITHDEBINFO ${__CUTLASS_CUDA_FLAGS_RELWITHDEBINFO} ${__CUTLASS_CUDA_NVCC_FLAGS_RELWITHDEBINFO})
    set(_FLAGS_DEBUG ${__CUTLASS_CUDA_FLAGS_DEBUG} ${__CUTLASS_CUDA_NVCC_FLAGS_DEBUG})
  endif()

  target_link_libraries(${TARGET} PRIVATE CUTLASS)

  target_compile_options(
    ${TARGET}
    PRIVATE
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:${_FLAGS}>
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:RELEASE>:${_FLAGS_RELEASE}>>
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:RELWITHDEBINFO>:${_FLAGS_RELWITHDEBINFO}>>
    $<$<COMPILE_LANGUAGE:${CUDA_COMPILE_LANGUAGE}>:$<$<CONFIG:DEBUG>:${_FLAGS_DEBUG}>>
    )

endfunction()

#
# The following items should eventually be pushed into cutlass/CMakeLists.txt
#

# GLOB for CUTLASS header files. Should we use a static list instead?
file(GLOB_RECURSE CUTLASS_INCLUDE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/cutlass/*.h)
file(GLOB_RECURSE CUTLASS_CUTLASS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/cutlass/*.h)
file(GLOB_RECURSE CUTLASS_NVRTC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/test test/unit/nvrtc/kernel/*.h)

###################################################################################################
#
# Define build targets
#
###################################################################################################

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include REGULAR_EXPRESSION ".*\.h")

add_library(CUTLASS INTERFACE)
add_library(nvidia::cutlass::cutlass ALIAS CUTLASS)
set_target_properties(CUTLASS PROPERTIES EXPORT_NAME cutlass)

set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "CUTLASS Header Library")

set(CUTLASS_GENERATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/library CACHE INTERNAL "Location of generator scripts")

# The following utility directory is needed even if the tools build is disabled, so it exists here.
set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include CACHE INTERNAL "")

include_directories(${CUTLASS_INCLUDE_DIR})

target_compile_features(CUTLASS INTERFACE cxx_std_11)
target_compile_definitions(CUTLASS INTERFACE CUTLASS_NAMESPACE=${CUTLASS_NAMESPACE})

if (NOT DEFINED CUTLASS_REVISION)

  find_package(Git QUIET)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    RESULT_VARIABLE CUTLASS_REVISION_RESULT
    OUTPUT_VARIABLE CUTLASS_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (CUTLASS_REVISION_RESULT)
    message(STATUS "CUTLASS Revision: Unable to detect, Git returned code ${CUTLASS_REVISION_RESULT}.")
  else()
    message(STATUS "CUTLASS Revision: ${CUTLASS_REVISION}")
  endif()

endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cutlass/version.h
  @ONLY)

target_include_directories(
  CUTLASS
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CUTLASS_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CUDA_TOOLKIT_ROOT_DIR}/include>
  )

install(
  DIRECTORY
  ${CUTLASS_INCLUDE_DIR}/
  ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  TARGETS CUTLASS
  EXPORT NvidiaCutlass
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

################################################################################

# Doxygen is available. Generate documentation
if (DOXYGEN_FOUND)
    # DOT is available. Enable graph generation in the documentation
    if (DOXYGEN_DOT_EXECUTABLE)
        set(CUTLASS_ENABLE_DOXYGEN_DOT ON CACHE BOOL "Use dot to generate graphs in the doxygen documentation.")
    else()
        set(CUTLASS_ENABLE_DOXYGEN_DOT OFF CACHE BOOL "Use dot to generate graphs in the doxygen documentation." FORCE)
    endif()

    if (CUTLASS_ENABLE_DOXYGEN_DOT)
        set(HAVE_DOT "YES")
    else()
        set(HAVE_DOT "NO")
    endif()

    # Add custom target for Doxygen.
    add_custom_target(cutlass_docs ${CMAKE_COMMAND} -E env
        "DOT_PATH=${DOXYGEN_DOT_EXECUTABLE}"
        "HAVE_DOT=${HAVE_DOT}"
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )
endif()

if(NOT WIN32)
  # Add common library search paths so executables and libraries can load and run
  # without LD_LIBRARY_PATH being set.
  link_libraries(
    "-Wl,-rpath,'$ORIGIN'"
    "-Wl,-rpath,'$ORIGIN/../lib64'"
    "-Wl,-rpath,'$ORIGIN/../lib'"
    "-Wl,-rpath,'${CUDA_TOOLKIT_ROOT_DIR}/lib64'"
    "-Wl,-rpath,'${CUDA_TOOLKIT_ROOT_DIR}/lib'"
    )
endif()

################################################################################

include(CTest)
enable_testing()
if (NOT TARGET test_all)
  add_custom_target(test_all)
endif()

set(CUTLASS_INSTALL_TESTS ON CACHE BOOL "Install test executables")
set(CUTLASS_TEST_EXECUTION_ENVIRONMENT "" CACHE BOOL "Environment in which to invoke unit test executables")

set(CMAKE_TEST_INSTALL_PREFIX test CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_PREFIX ${CMAKE_TEST_INSTALL_PREFIX}/cutlass CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_BINDIR ${CUTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_LIBDIR ${CUTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")

install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_LIBDIR})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX}/ctest)

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cuBLAS.cmake)

if (CUTLASS_ENABLE_CUBLAS)
  target_compile_definitions(CUTLASS INTERFACE CUTLASS_ENABLE_CUBLAS=1)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cuDNN.cmake)

if (CUTLASS_ENABLE_CUDNN)
  target_compile_definitions(CUTLASS INTERFACE CUTLASS_ENABLE_CUDNN=1)
endif()

################################################################################

set(CUTLASS_CTEST_TEMPLATE_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestTestfile.config.cmake)
set(CUTLASS_CTEST_GENERATED_FILES "" CACHE INTERNAL "")

function(cutlass_add_executable_tests NAME TARGET)
# 
# Generates test rules for `make test`, `make test_all`, and `ctest` invoked from either the 
# <CMAKE_BINARY_DIR> or the <CMAKE_INSTALL_PREFIX>/<CUTLASS_TEST_INSTALL_PREFIX> after installation.
# 
# NAME: The base name for the test. Can be run with `make <NAME>` or `ctest -R 'c<NAME>'`.
# TARGET: The target corresponding to the executable under test.
# DISABLE_EXECUTABLE_INSTALL_RULE: An option, if given, that disables creating an install rule for TARGET.
# DEPENDS: A list of targets or files on which this test is dependent.
# DEPENDEES: A list of targets which should depend on this test.
# TEST_COMMAND_OPTIONS: A list of variables (i.e. by reference params) which contain command line arguments
#   to pass to the test executable. A unique test with suffix _0, _1, ... is generated for each set of 
#   options given. If this option is not used, a single test with no arguments is generated.
# RESULT_CACHE_FILE: A file to be installed alongside the test executable with pre-computed
#   test results to speed up test runtime.
# 

  set(options DISABLE_EXECUTABLE_INSTALL_RULE)
  set(oneValueArgs DISABLE_TESTS RESULT_CACHE_FILE)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  if (__RESULT_CACHE_FILE)

    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${__RESULT_CACHE_FILE} "$<TARGET_FILE_DIR:${TARGET}>"
      )

  endif()

  if (NOT __DISABLE_EXECUTABLE_INSTALL_RULE AND CUTLASS_INSTALL_TESTS)
  
    # file(RELATIVE_PATH CMAKE_CURRENT_BINARY_RELATIVE_DIR ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  
    install(
      TARGETS ${TARGET}
      RUNTIME DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR}
      )

    if (__RESULT_CACHE_FILE)

     install(
       FILES ${__RESULT_CACHE_FILE}
       DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR}/
       )

    endif()
  
  endif()

  if (NOT __TEST_COMMAND_OPTIONS)
    set(__TEST_COMMAND_OPTIONS " ")
  endif()

  list(LENGTH __TEST_COMMAND_OPTIONS CMD_COUNT)
  set(CMD_IDX 0)

  if (CMD_COUNT GREATER 1)
    add_custom_target(${NAME} DEPENDS ${TARGET} ${__DEPENDS})
    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${NAME})
    endforeach()
  endif()

  foreach(CMD_OPTIONS ${__TEST_COMMAND_OPTIONS})

    if (CMD_COUNT GREATER 1)
      set(TEST_NAME ${NAME}_${CMD_IDX})
    else()
      set(TEST_NAME ${NAME})
    endif()

    # The following rigmarole is needed to deal with spaces and possible quotes in 
    # command line arguments. The options are passed "by reference" as the actual
    # variable names holding the real options. We then expand these in a way that
    # preserves any quotes. Note, they have to be in this order for it to work for 
    # all the use cases below.

    set(CMD_OPTIONS ${${CMD_OPTIONS}})
    list(JOIN CMD_OPTIONS " " TEST_COMMAND_OPTIONS)
    separate_arguments(CMD_OPTIONS)
   
    add_custom_target(
      ${TEST_NAME}
      COMMAND
      ${CUTLASS_TEST_EXECUTION_ENVIRONMENT} $<TARGET_FILE:${TARGET}> ${CMD_OPTIONS}
      DEPENDS
      ${TARGET}
      )

    if (CMD_COUNT GREATER 1)
      add_dependencies(${NAME} ${TEST_NAME})
    endif()

    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${TEST_NAME})
    endforeach()

    add_test(
      NAME c${TEST_NAME}
      COMMAND ${CUTLASS_TEST_EXECUTION_ENVIRONMENT} $<TARGET_FILE:${TARGET}> ${CMD_OPTIONS}
      )

    set_tests_properties(c${TEST_NAME} PROPERTIES DISABLED ${__DISABLE_TESTS})

    if (CUTLASS_INSTALL_TESTS)

      # To run the tests from an install package with tests enabled, we need to generate test files
      # that don't rely on the current directory structure in build.  

      set(TEST_NAME c${TEST_NAME})
      set(TEST_EXE $<TARGET_FILE_NAME:${TARGET}>)
      set(TEST_EXE_WORKING_DIRECTORY ./${CMAKE_INSTALL_BINDIR})
      configure_file("${CUTLASS_CTEST_TEMPLATE_FILE}" "${CMAKE_PROJECT_DIR}${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.${TEST_NAME}.config.cmake" @ONLY)

      file(GENERATE 
        OUTPUT "${CMAKE_PROJECT_DIR}${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.${TEST_NAME}.cmake" 
        INPUT "${CMAKE_PROJECT_DIR}${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.${TEST_NAME}.config.cmake"
        )
  
      install(
        FILES "${CMAKE_PROJECT_DIR}${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.${TEST_NAME}.cmake"
        DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX}/ctest/
        )
  
      set(CUTLASS_CTEST_GENERATED_FILES ${CUTLASS_CTEST_GENERATED_FILES};ctest/CTestTestfile.${TEST_NAME}.cmake CACHE INTERNAL "")
  
    endif()

    math(EXPR CMD_IDX "${CMD_IDX} + 1")

  endforeach()

endfunction()

if (CUTLASS_ENABLE_TOOLS)
  add_subdirectory(tools)
  if (CUTLASS_ENABLE_PROFILER)
    add_dependencies(test_all test_profiler)
  endif()  
endif()
if (CUTLASS_ENABLE_EXAMPLES)
  add_subdirectory(examples)
  add_dependencies(test_all test_examples)
endif()

if (CUTLASS_ENABLE_TESTS)
  add_subdirectory(test)
  add_dependencies(test_all test_unit)
endif()

if (CUTLASS_INSTALL_TESTS)

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/cmake")

  file(WRITE "${CMAKE_BINARY_DIR}/cmake/CTestTestfile.cmake" "# Generated File\n")
  foreach(GENERATED_FILE ${CUTLASS_CTEST_GENERATED_FILES})
    file(APPEND "${CMAKE_BINARY_DIR}/cmake/CTestTestfile.cmake" "include(${GENERATED_FILE})\n")
  endforeach()

  install(
    FILES "${CMAKE_BINARY_DIR}/cmake/CTestTestfile.cmake"
    DESTINATION "${CUTLASS_TEST_INSTALL_PREFIX}/"
    )

endif()

#? install(
#?   FILES ${CMAKE_BINARY_DIR}/CTestTestfile.cmake
#?   DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX}/
#?   )
#? 
#? install(
#?   DIRECTORY 
#?     ${CMAKE_BINARY_DIR}/tools
#?     ${CMAKE_BINARY_DIR}/test
#?   DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX}/
#?   FILES_MATCHING PATTERN "CTestTestfile.cmake"
#?   )

################################################################################

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NvidiaCutlassConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
  )

install(
  EXPORT NvidiaCutlass
  NAMESPACE nvidia::cutlass::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
  FILE NvidiaCutlassTargets.cmake
  )

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/NvidiaCutlassPackageConfig.cmake)
