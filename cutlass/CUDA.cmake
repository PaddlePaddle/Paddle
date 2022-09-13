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

if(CUDA_COMPILER MATCHES "[Cc]lang")
  set(CUTLASS_NATIVE_CUDA_INIT ON)
elseif(CMAKE_VERSION VERSION_LESS 3.12.4)
  set(CUTLASS_NATIVE_CUDA_INIT OFF)
else()
  set(CUTLASS_NATIVE_CUDA_INIT ON)
endif()

set(CUTLASS_NATIVE_CUDA ${CUTLASS_NATIVE_CUDA_INIT} CACHE BOOL "Utilize the CMake native CUDA flow")

if(NOT DEFINED ENV{CUDACXX} AND NOT DEFINED ENV{CUDA_BIN_PATH} AND DEFINED ENV{CUDA_PATH})
  # For backward compatibility, allow use of CUDA_PATH.
  set(ENV{CUDACXX} $ENV{CUDA_PATH}/bin/nvcc)
endif()

if(CUTLASS_NATIVE_CUDA)

  enable_language(CUDA)

  if(NOT CUDA_VERSION)
    set(CUDA_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
  endif()
  if(NOT CUDA_TOOLKIT_ROOT_DIR)
    get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
  endif()

else()

  find_package(CUDA REQUIRED)
  # We workaround missing variables with the native flow by also finding the CUDA toolkit the old way.

  if(NOT CMAKE_CUDA_COMPILER_VERSION)
    set(CMAKE_CUDA_COMPILER_VERSION ${CUDA_VERSION})
  endif()

endif()

if (CUDA_VERSION VERSION_LESS 9.2)
  message(FATAL_ERROR "CUDA 9.2+ Required, Found ${CUDA_VERSION}.")
endif()
if(NOT CUTLASS_NATIVE_CUDA OR CUDA_COMPILER MATCHES "[Cc]lang")
  set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
  message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
endif()

find_library(
  CUDART_LIBRARY cudart
  PATHS
  ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES
  lib/x64
  lib64
  lib
  NO_DEFAULT_PATH
  # We aren't going to search any system paths. We want to find the runtime
  # in the CUDA toolkit we're building against.
  )

if(NOT TARGET cudart AND CUDART_LIBRARY)

  message(STATUS "CUDART: ${CUDART_LIBRARY}")

  if(WIN32)
    add_library(cudart STATIC IMPORTED GLOBAL)
    # Even though we're linking against a .dll, in Windows you statically link against
    # the .lib file found under lib/x64. The .dll will be loaded at runtime automatically
    # from the PATH search.
  else()
    add_library(cudart SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::cudart ALIAS cudart)

  set_property(
    TARGET cudart
    PROPERTY IMPORTED_LOCATION
    ${CUDART_LIBRARY}
    )

elseif(TARGET cudart)

  message(STATUS "CUDART: Already Found")

else()

  message(STATUS "CUDART: Not Found")

endif()

find_library(
  CUDA_DRIVER_LIBRARY cuda
  PATHS
  ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES
  lib/x64
  lib64
  lib
  lib64/stubs
  lib/stubs
  NO_DEFAULT_PATH
  # We aren't going to search any system paths. We want to find the runtime
  # in the CUDA toolkit we're building against.
  )

if(NOT TARGET cuda_driver AND CUDA_DRIVER_LIBRARY)

  message(STATUS "CUDA Driver: ${CUDA_DRIVER_LIBRARY}")

  if(WIN32)
    add_library(cuda_driver STATIC IMPORTED GLOBAL)
    # Even though we're linking against a .dll, in Windows you statically link against
    # the .lib file found under lib/x64. The .dll will be loaded at runtime automatically
    # from the PATH search.
  else()
    add_library(cuda_driver SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::cuda_driver ALIAS cuda_driver)

  set_property(
    TARGET cuda_driver
    PROPERTY IMPORTED_LOCATION
    ${CUDA_DRIVER_LIBRARY}
    )

elseif(TARGET cuda_driver)

  message(STATUS "CUDA Driver: Already Found")

else()

  message(STATUS "CUDA Driver: Not Found")

endif()

find_library(
  NVRTC_LIBRARY nvrtc
  PATHS
  ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES
  lib/x64
  lib64
  lib
  NO_DEFAULT_PATH
  # We aren't going to search any system paths. We want to find the runtime
  # in the CUDA toolkit we're building against.
  )

if(NOT TARGET nvrtc AND NVRTC_LIBRARY)

  message(STATUS "NVRTC: ${NVRTC_LIBRARY}")

  if(WIN32)
    add_library(nvrtc STATIC IMPORTED GLOBAL)
    # Even though we're linking against a .dll, in Windows you statically link against
    # the .lib file found under lib/x64. The .dll will be loaded at runtime automatically
    # from the PATH search.
  else()
    add_library(nvrtc SHARED IMPORTED GLOBAL)
  endif()

  add_library(nvidia::nvrtc ALIAS nvrtc)

  set_property(
    TARGET nvrtc
    PROPERTY IMPORTED_LOCATION
    ${NVRTC_LIBRARY}
    )

elseif(TARGET nvrtc)

  message(STATUS "NVRTC: Already Found")

else()

  message(STATUS "NVRTC: Not Found")

endif()

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
# Some platforms (e.g. Visual Studio) don't add the CUDA include directories to the system include
# paths by default, so we add it explicitly here.

function(cutlass_correct_source_file_language_property)
  if(CUDA_COMPILER MATCHES "[Cc]lang")
    foreach(File ${ARGN})
      if(File MATCHES ".*\.cu$")
        set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
      endif()
    endforeach()
  endif()
endfunction()

if (MSVC OR CUTLASS_LIBRARY_KERNELS MATCHES "all")
  set(CUTLASS_UNITY_BUILD_ENABLED_INIT ON)
else()
  set(CUTLASS_UNITY_BUILD_ENABLED_INIT OFF)
endif()

set(CUTLASS_UNITY_BUILD_ENABLED ${CUTLASS_UNITY_BUILD_ENABLED_INIT} CACHE BOOL "Enable combined source compilation")
set(CUTLASS_UNITY_BUILD_BATCH_SIZE 16 CACHE STRING "Batch size for unified source files")

function(cutlass_unify_source_files TARGET_ARGS_VAR)

  set(options)
  set(oneValueArgs BATCH_SOURCES BATCH_SIZE)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED TARGET_ARGS_VAR)
    message(FATAL_ERROR "TARGET_ARGS_VAR parameter is required")
  endif()

  if (__BATCH_SOURCES AND NOT DEFINED __BATCH_SIZE)
    set(__BATCH_SIZE ${CUTLASS_UNITY_BUILD_BATCH_SIZE})
  endif()

  if (CUTLASS_UNITY_BUILD_ENABLED AND DEFINED __BATCH_SIZE AND __BATCH_SIZE GREATER 1)

    set(CUDA_FILE_ARGS)
    set(TARGET_SOURCE_ARGS)

    foreach(ARG ${__UNPARSED_ARGUMENTS})
      if(${ARG} MATCHES ".*\.cu$")
        list(APPEND CUDA_FILE_ARGS ${ARG})
      else()
        list(APPEND TARGET_SOURCE_ARGS ${ARG})
      endif()
    endforeach()

    list(LENGTH CUDA_FILE_ARGS NUM_CUDA_FILE_ARGS)
    while(NUM_CUDA_FILE_ARGS GREATER 0)
      list(SUBLIST CUDA_FILE_ARGS 0 ${__BATCH_SIZE} CUDA_FILE_BATCH)
      string(SHA256 CUDA_FILE_BATCH_HASH "${CUDA_FILE_BATCH}")
      string(SUBSTRING ${CUDA_FILE_BATCH_HASH} 0 12 CUDA_FILE_BATCH_HASH)
      set(BATCH_FILE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.unity.${CUDA_FILE_BATCH_HASH}.cu)
      message(STATUS "Generating ${BATCH_FILE}")
      file(WRITE ${BATCH_FILE} "// Unity File - Auto Generated!\n")
      foreach(CUDA_FILE ${CUDA_FILE_BATCH})
        get_filename_component(CUDA_FILE_ABS_PATH ${CUDA_FILE} ABSOLUTE)
        file(APPEND ${BATCH_FILE} "#include \"${CUDA_FILE_ABS_PATH}\"\n")
      endforeach()
      list(APPEND TARGET_SOURCE_ARGS ${BATCH_FILE})
      if (NUM_CUDA_FILE_ARGS LESS_EQUAL __BATCH_SIZE)
        break()
      endif()
      list(SUBLIST CUDA_FILE_ARGS ${__BATCH_SIZE} -1 CUDA_FILE_ARGS)
      list(LENGTH CUDA_FILE_ARGS NUM_CUDA_FILE_ARGS)
    endwhile()

  else()

    set(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  endif()

  set(${TARGET_ARGS_VAR} ${TARGET_SOURCE_ARGS} PARENT_SCOPE)

endfunction()
function(cutlass_add_library NAME)

  set(options)
  set(oneValueArgs EXPORT_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  if(CUTLASS_NATIVE_CUDA OR CUDA_COMPILER MATCHES "clang")
    cutlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
    add_library(${NAME} ${TARGET_SOURCE_ARGS})
  else()
    set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
    cuda_add_library(${NAME} ${TARGET_SOURCE_ARGS})
  endif()

  cutlass_apply_standard_compile_options(${NAME})
  cutlass_apply_cuda_gencode_flags(${NAME})

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

  if(__EXPORT_NAME)
    add_library(nvidia::cutlass::${__EXPORT_NAME} ALIAS ${NAME})
    set_target_properties(${NAME} PROPERTIES EXPORT_NAME ${__EXPORT_NAME})
  endif()

endfunction()

function(cutlass_add_executable NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  if(CUTLASS_NATIVE_CUDA OR CUDA_COMPILER MATCHES "clang")
    cutlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
    add_executable(${NAME} ${TARGET_SOURCE_ARGS})
  else()
    set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
    cuda_add_executable(${NAME} ${TARGET_SOURCE_ARGS})
  endif()

  cutlass_apply_standard_compile_options(${NAME})
  cutlass_apply_cuda_gencode_flags(${NAME})

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

endfunction()

function(cutlass_target_sources NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})
  cutlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  target_sources(${NAME} ${TARGET_SOURCE_ARGS})

endfunction()
