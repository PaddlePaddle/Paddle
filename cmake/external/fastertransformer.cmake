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
set(FASTER_TRANSFORMER_PATH
    "${THIRD_PARTY_PATH}/fastertransformer"
    CACHE PATH "FASTER_TRANSFORMER_PATH" FORCE)
set(FASTER_TRANSFORMER_PREFIX_DIR ${FASTER_TRANSFORMER_PATH})
set(FASTER_TRANSFORMER_SOURCE_DIR
    ${FASTER_TRANSFORMER_PREFIX_DIR}/src/extern_fastertransformer)
set(FASTER_TRANSFORMER_INSTALL_DIR
    ${THIRD_PARTY_PATH}/install/fastertransformer)

set(FASTER_TRANSFORMER_REPOSITORY ${GIT_URL}/NVIDIA/FasterTransformer.git)

set(FASTER_TRANSFORMER_TAG release/v5.1_bugfix_tag)

set(FASTER_TRANSFORMER_INCLUDE_DIR
    ${FASTER_TRANSFORMER_PREFIX_DIR}/src/extern_fastertransformer
    CACHE PATH "FASTER_TRANSFORMER_INCLUDE_DIR" FORCE)

set(FASTER_TRANSFORMER_TRT_FUSED_MHA_LIBRARIES
    ${FASTER_TRANSFORMER_INSTALL_DIR}/lib/libtrt_fused_multi_head_attention.a)

set(BUILD_COMMAND ${CMAKE_COMMAND} --build . --target
                  trt_fused_multi_head_attention)
set(INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ./lib
                    ${FASTER_TRANSFORMER_INSTALL_DIR}/lib)

message("FASTER_TRANSFORMER_INCLUDE_DIR is ${FASTER_TRANSFORMER_INCLUDE_DIR}")

include_directories(${FASTER_TRANSFORMER_INCLUDE_DIR})

set(FASTER_TRANSFORMER_OPTIONAL_ARGS -DCMAKE_BUILD_TYPE=Release -DBUILD_TRT=ON)

ExternalProject_Add(
  extern_fastertransformer
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${FASTER_TRANSFORMER_PREFIX_DIR}
  DOWNLOAD_DIR ${FASTER_TRANSFORMER_SOURCE_DIR}
  DOWNLOAD_COMMAND
  COMMAND rm -rf .git cmake CMakeLists.txt 3rdparty src
  COMMAND git init
  COMMAND git remote add origin ${FASTER_TRANSFORMER_REPOSITORY}
  COMMAND git config core.sparsecheckout true
  COMMAND echo /3rdparty/CMakeLists.txt >> .git/info/sparse-checkout
  COMMAND echo /3rdparty/INIReader.h >> .git/info/sparse-checkout
  COMMAND echo /3rdparty/trt_fused_multihead_attention >>
          .git/info/sparse-checkout
  COMMAND echo /cmake/ >> .git/info/sparse-checkout
  COMMAND echo /src/fastertransformer/utils/cuda_utils.h >>
          .git/info/sparse-checkout
  COMMAND echo /src/fastertransformer/utils/cuda_bf16_wrapper.h >>
          .git/info/sparse-checkout
  COMMAND echo /src/fastertransformer/utils/logger.h >>
          .git/info/sparse-checkout
  COMMAND echo /src/fastertransformer/utils/string_utils.h >>
          .git/info/sparse-checkout
  COMMAND git fetch --depth 1 origin ${FASTER_TRANSFORMER_TAG}
  COMMAND git checkout FETCH_HEAD
  PATCH_COMMAND
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    "${PADDLE_SOURCE_DIR}/patches/fastertransformer/CMakeLists.txt"
    "<SOURCE_DIR>/"
  BUILD_COMMAND ${BUILD_COMMAND}
  INSTALL_COMMAND ${INSTALL_COMMAND}
  CMAKE_ARGS ${FASTER_TRANSFORMER_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FASTER_TRANSFORMER_INSTALL_DIR}
                   -DCMAKE_BUILD_TYPE:STRING=Release
  BUILD_BYPRODUCTS ${FASTER_TRANSFORMER_TRT_FUSED_MHA_LIBRARIES})

add_definitions(-DFASTERTRANSFORMER_TRT_FUSED_MHA_AVALIABLE)
add_library(fastertransformer_trt_fused_mha STATIC IMPORTED GLOBAL)
set_property(
  TARGET fastertransformer_trt_fused_mha
  PROPERTY IMPORTED_LOCATION ${FASTER_TRANSFORMER_TRT_FUSED_MHA_LIBRARIES})
add_dependencies(fastertransformer_trt_fused_mha extern_fastertransformer)
set(WITH_FASTERTRANSFORMER_MHA
    ON
    CACHE BOOL "Enable fastertransformer mha." FORCE)
