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
set(FMHA_PATH
    "${THIRD_PARTY_PATH}/extern_fmha"
    CACHE PATH "FMHA_PATH" FORCE)
set(FMHA_PREFIX_DIR ${FMHA_PATH})
set(FMHA_SOURCE_DIR ${FMHA_PREFIX_DIR}/src/extern_fmha)
set(FMHA_INSTALL_DIR ${THIRD_PARTY_PATH}/install/extern_fmha)

set(FMHA_REPOSITORY ${GIT_URL}/wwbitejotunn/FasterTransformer.git)

set(FMHA_TAG 3rd_party_fmha)

set(FMHA_INCLUDE_DIR
    ${FMHA_PREFIX_DIR}/src/extern_fmha
    CACHE PATH "FMHA_INCLUDE_DIR" FORCE)

set(FMHA_FUSED_MHA_LIBRARIES
    ${FMHA_INSTALL_DIR}/lib/libtrt_fused_multi_head_attention.a)

set(BUILD_COMMAND ${CMAKE_COMMAND} --build . --target
                  trt_fused_multi_head_attention)
set(INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ./lib
                    ${FMHA_INSTALL_DIR}/lib)

message("FMHA_INCLUDE_DIR is ${FMHA_INCLUDE_DIR}")

include_directories(${FMHA_INCLUDE_DIR})

set(FMHA_OPTIONAL_ARGS -DCMAKE_BUILD_TYPE=Release -DBUILD_TRT=ON)

ExternalProject_Add(
  extern_fmha
  GIT_REPOSITORY ${FMHA_REPOSITORY}
  GIT_TAG ${FMHA_TAG}
  PREFIX ${FMHA_PREFIX_DIR}
  BUILD_COMMAND ${BUILD_COMMAND}
  INSTALL_COMMAND ${INSTALL_COMMAND}
  CMAKE_ARGS ${FMHA_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FMHA_INSTALL_DIR}
                   -DCMAKE_BUILD_TYPE:STRING=Release
  BUILD_BYPRODUCTS ${FMHA_FUSED_MHA_LIBRARIES})

add_definitions(-DTRT_FUSED_MHA_AVALIABLE)
add_library(trt_fused_mha STATIC IMPORTED GLOBAL)
set_property(TARGET trt_fused_mha PROPERTY IMPORTED_LOCATION
                                           ${FMHA_FUSED_MHA_LIBRARIES})
add_dependencies(trt_fused_mha extern_fmha)
set(WITH_FUSED_MHA
    ON
    CACHE BOOL "Enable fushed mha." FORCE)
