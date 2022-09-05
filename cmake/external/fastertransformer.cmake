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

if(NOT WITH_TENSORRT)
  return()
endif()

if(WITH_ARM OR WIN32)
  message(SEND_ERROR "The current FasterTransformer support linux only")
  return()
endif()

include(ExternalProject)

set(FASTER_TRANSFORMER_PATH
    "${THIRD_PARTY_PATH}/fastertransformer"
    CACHE PATH "FASTER_TRANSFORMER_PATH" FORCE)
set(FASTER_TRANSFORMER_PREFIX_DIR ${FASTER_TRANSFORMER_PATH})

set(FASTER_TRANSFORMER_INSTALL_DIR ${THIRD_PARTY_PATH}/install/fastertransformer)

set(FASTER_TRANSFORMER_REPOSITORY ${GIT_URL}/NVIDIA/FasterTransformer.git)

set(FASTER_TRANSFORMER_TAG release/v5.1_bugfix_tag)

set(FASTER_TRANSFORMER_INCLUDE_DIR ${FASTER_TRANSFORMER_PREFIX_DIR}/src/extern_fastertransformer
    CACHE PATH "FASTER_TRANSFORMER_INCLUDE_DIR" FORCE)

set(FASTER_TRANSFORMER_TRT_FUSED_MHA_LIBRARIES
    ${FASTER_TRANSFORMER_INSTALL_DIR}/lib/libtrt_fused_multi_head_attention.a)


set(BUILD_COMMAND ${CMAKE_COMMAND} --build . --target trt_fused_multi_head_attention)
set(INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ./lib ${FASTER_TRANSFORMER_INSTALL_DIR}/lib)

message("FASTER_TRANSFORMER_INCLUDE_DIR is ${FASTER_TRANSFORMER_INCLUDE_DIR}")

include_directories(${FASTER_TRANSFORMER_INCLUDE_DIR})

set(FASTER_TRANSFORMER_OPTIONAL_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_TRT=ON)

ExternalProject_Add(
    extern_fastertransformer
    ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
    GIT_REPOSITORY ${FASTER_TRANSFORMER_REPOSITORY}
    GIT_TAG ${FASTER_TRANSFORMER_TAG}
    PREFIX ${FASTER_TRANSFORMER_PREFIX_DIR}
    UPDATE_COMMAND ""
    BUILD_COMMAND ${BUILD_COMMAND} 
    INSTALL_COMMAND ${INSTALL_COMMAND}
    CMAKE_ARGS  ${FASTER_TRANSFORMER_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS 
        -DCMAKE_INSTALL_PREFIX:PATH=${FASTER_TRANSFORMER_INSTALL_DIR}
        -DCMAKE_BUILD_TYPE:STRING=Release
    BUILD_BYPRODUCTS ${FASTER_TRANSFORMER_TRT_FUSED_MHA_LIBRARIES})

add_library(fastertransformer_trt_fused_mha STATIC IMPORTED GLOBAL)
set_property(TARGET fastertransformer_trt_fused_mha PROPERTY IMPORTED_LOCATION ${FASTER_TRANSFORMER_TRT_FUSED_MHA_LIBRARIES})
add_dependencies(fastertransformer_trt_fused_mha extern_fastertransformer)