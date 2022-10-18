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

set(CUDNN_FRONTEND_CUDNN_MIN_VERSION 8400)

if(NOT WITH_GPU)
  message(FATAL_ERROR "Can't enable CUDNN Frontend API without CUDA.")
endif()
if(CUDNN_VERSION LESS 8400)
  message(
    FATAL_ERROR
      "Minimum CUDNN version is ${CUDNN_FRONTEND_CUDNN_MIN_VERSION}. Current: ${CUDNN_VERSION}"
  )
endif()

message(STATUS "Adding cudnn-frontend.")

# Version: v0.7.1
set(CUDNN_FRONTEND_PREFIX_DIR ${THIRD_PARTY_PATH}/cudnn-frontend)
set(CUDNN_FRONTEND_SOURCE_DIR
    ${THIRD_PARTY_PATH}/cudnn-frontend/src/extern_cudnn_frontend/include)
set(CUDNN_FRONTEND_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git)
set(CUDNN_FRONTEND_TAG v0.7.1)

set(CUDNN_FRONTEND_INCLUDE_DIR ${CUDNN_FRONTEND_SOURCE_DIR})
include_directories(${CUDNN_FRONTEND_INCLUDE_DIR})

ExternalProject_Add(
  extern_cudnn_frontend
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${CUDNN_FRONTEND_REPOSITORY}
  GIT_TAG ${CUDNN_FRONTEND_TAG}
  PREFIX ${CUDNN_FRONTEND_PREFIX_DIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND sed -i s/NULL/nullptr/g
                ${CUDNN_FRONTEND_SOURCE_DIR}/cudnn_frontend_ExecutionPlan.h
  COMMAND sed -i s/NULL/nullptr/g
          ${CUDNN_FRONTEND_SOURCE_DIR}/cudnn_frontend_OperationGraph.h
  COMMAND sed -i "s/ ::cudnn/ cudnn/g"
          ${CUDNN_FRONTEND_SOURCE_DIR}/cudnn_frontend_find_plan.h
  COMMAND sed -i "s/^\\(auto\\)/inline \\1/g"
          ${CUDNN_FRONTEND_SOURCE_DIR}/cudnn_frontend_get_plan.h
  COMMAND sed -i "s/, cudnn_frontend::ExecutionPlanCache_v1::compare//"
          ${CUDNN_FRONTEND_SOURCE_DIR}/cudnn_frontend_ExecutionPlanCache.h
  COMMAND
    patch -d ${CUDNN_FRONTEND_SOURCE_DIR} -p2 <
    ${PADDLE_SOURCE_DIR}/patches/cudnn-frontend/cudnn_frontend_ExecutionPlan_patch_1.patch
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
  TIMEOUT $ENV{DOWNLOAD_TIMEOUT})

add_library(cudnn-frontend INTERFACE)
add_dependencies(cudnn-frontend extern_cudnn_frontend)
