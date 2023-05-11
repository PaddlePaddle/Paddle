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

set(CUTLASS_PREFIX_DIR ${THIRD_PARTY_PATH}/cutlass)

set(CUTLASS_REPOSITORY https://github.com/NVIDIA/cutlass.git)
set(CUTLASS_TAG v2.11.0)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/cutlass)
include_directories("${THIRD_PARTY_PATH}/cutlass/src/extern_cutlass/")
include_directories("${THIRD_PARTY_PATH}/cutlass/src/extern_cutlass/include/")
include_directories(
  "${THIRD_PARTY_PATH}/cutlass/src/extern_cutlass/tools/util/include/")

add_definitions("-DPADDLE_WITH_CUTLASS")

if(NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp REQUIRED)
endif()

ExternalProject_Add(
  extern_cutlass
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${CUTLASS_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_custom_target(
  cutlass_codegen
  COMMAND
    rm -rf
    ${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator/build
  COMMAND
    mkdir -p
    ${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator/build/generated/gemm
  COMMAND
    ${PYTHON_EXECUTABLE} -B
    ${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator/gather_gemm_scatter_generator.py
    "${THIRD_PARTY_PATH}/cutlass/src/extern_cutlass/tools/library/scripts/"
    "${CMAKE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/cutlass_generator/build"
    "${CMAKE_CUDA_COMPILER_VERSION}"
  VERBATIM)

add_library(cutlass INTERFACE)

add_dependencies(cutlass_codegen extern_cutlass)
add_dependencies(cutlass extern_cutlass)
