# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

include(ExternalProject)

if(NOT DEFINED ENV{MACA_PATH})
    MESSAGE(FATAL_ERROR "MACA_PATH undefined, mctlass not found!")
else()
    # set(CUTLASS_PREFIX_DIR "$ENV{MACA_PATH}/include/mctlass")
    set(CUTLASS_SOURCE_DIR "$ENV{MACA_PATH}/include/mctlass")
endif()
if(NOT EXISTS ${CUTLASS_SOURCE_DIR})
    MESSAGE(FATAL_ERROR "mctlass not found in MACA_PATH!")
endif()
# PATCH FOR MACA
set(CUTLASS_PATCH_DIR ${PADDLE_SOURCE_DIR}/third_party/cutlass)
set(CUTLASS_PREFIX_DIR ${PADDLE_SOURCE_DIR}/third_party/cutlass)
# set(CUTLASS_TAG v2.11.0)
# set(CUTLASS_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/cutlass)
include_directories("${CUTLASS_SOURCE_DIR}/")
include_directories("${CUTLASS_SOURCE_DIR}/include/")
# include_directories("${CUTLASS_SOURCE_DIR}/tools/util/include/")

add_definitions("-DPADDLE_WITH_CUTLASS")
add_definitions("-DSPCONV_WITH_CUTLASS=0")

if(NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp REQUIRED)
endif()

ExternalProject_Add(
  extern_cutlass
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${CUTLASS_SOURCE_DIR}
  PREFIX ${CUTLASS_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

set(tmp_gemm_operations_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator/generated/gemm/all_gemm_operations.h.tmp
)
set(tmp_configurations_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator/generated/gemm/configurations.h.tmp
)
set(gemm_operations_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator/all_gemm_operations.h
)
set(configurations_file
    ${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator/configurations.h
)

add_custom_target(
  cutlass_codegen
  COMMAND
    ${PYTHON_EXECUTABLE} -B
    ${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator/gather_gemm_scatter_generator.py
    "${CUTLASS_PATCH_DIR}/tools/library/scripts/"
    "${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator"
    "${CMAKE_CUDA_COMPILER_VERSION}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_gemm_operations_file}
          ${gemm_operations_file}
  COMMAND
    ${CMAKE_COMMAND} -E echo
    "copy_if_different ${tmp_gemm_operations_file} to ${gemm_operations_file}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_configurations_file}
          ${configurations_file}
  COMMAND
    ${CMAKE_COMMAND} -E echo
    "copy_if_different ${tmp_configurations_file} to ${configurations_file}"
  VERBATIM)

add_library(cutlass INTERFACE)

add_dependencies(cutlass_codegen extern_cutlass)
add_dependencies(cutlass extern_cutlass)
