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

include(CTest)

cutlass_add_library(
  cutlass_test_unit_infra 
  OBJECT
  common/filter_architecture.cpp
  )

target_link_libraries(
  cutlass_test_unit_infra
  PUBLIC
  CUTLASS
  cutlass_tools_util_includes
  $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas>
  gtest
  cudart
  )

cutlass_add_library(
  cutlass_test_unit_infra_lib
  OBJECT
  test_unit.cpp
  )

target_link_libraries(
  cutlass_test_unit_infra_lib
  PUBLIC
  cutlass_test_unit_infra
  )

set(CUTLASS_TEST_UNIT_RESULTS_CACHE_DIR ${CMAKE_CURRENT_LIST_DIR}/data/hashes)

function(cutlass_test_unit_add_executable NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  cutlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS})

  target_compile_definitions(${NAME} PUBLIC CUTLASS_TARGET_NAME="${NAME}")

  target_link_libraries(
    ${NAME}
    PRIVATE
    cutlass_test_unit_infra
    cutlass_test_unit_infra_lib
  )

  string(REGEX REPLACE cutlass_ "" NAME_STEM ${NAME})

  set(RESULT_CACHE_FILE "${CUTLASS_TEST_UNIT_RESULTS_CACHE_DIR}/cached_results_${NAME}.txt")

  if (EXISTS ${RESULT_CACHE_FILE})
    set(RESULT_CACHE_FILE_ARGS RESULT_CACHE_FILE ${RESULT_CACHE_FILE})
  endif()

  set(CUTLASS_TEST_UNIT_TEST_COMMAND_OPTIONS --gtest_output=xml:${NAME_STEM}.gtest.xml)

  cutlass_add_executable_tests(
    ${NAME_STEM} ${NAME}
    TEST_COMMAND_OPTIONS CUTLASS_TEST_UNIT_TEST_COMMAND_OPTIONS
    ${RESULT_CACHE_FILE_ARGS}
    )

endfunction()

add_custom_target(cutlass_test_unit)
add_custom_target(test_unit)

set(SUBDIRS
  core
  gemm
  conv
  layout
  transform
  epilogue
  reduction
  util
  )

if(TARGET nvidia::nvrtc AND TARGET nvidia::cuda_driver)
  set(CUTLASS_NVRTC_ENABLE_INIT ON)
else()
  set(CUTLASS_NVRTC_ENABLE_INIT OFF)
endif()  

set(CUTLASS_NVRTC_ENABLE ${CUTLASS_NVRTC_ENABLE_INIT} CACHE BOOL "Enable NVRTC support")

if (CUTLASS_NVRTC_ENABLE)
  list(APPEND SUBDIRS nvrtc)
endif()

foreach(SUBDIR ${SUBDIRS})

  add_subdirectory(${SUBDIR})
  add_dependencies(cutlass_test_unit cutlass_test_unit_${SUBDIR})
  add_dependencies(test_unit test_unit_${SUBDIR})

endforeach()

