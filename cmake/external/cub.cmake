# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# extern_cub  has code __FILE_, If the path of extern_cub is changed,
# it will effect about 30+ cu files sccache hit and slow compile speed  on windows.
# Therefore, a fixed CUB_PATH will be input to increase the sccache hit rate.
set(CUB_PATH
    "${THIRD_PARTY_PATH}/cub"
    CACHE STRING "A path setting for external_cub path.")
set(CUB_PREFIX_DIR ${CUB_PATH})

set(CUB_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/cub)

if(${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL 11.6)
  # cuda_11.6/11.7/11.8‘s own cub is 1.15.0, which will cause compiling error in windows.
  set(CUB_TAG 1.16.0)
  execute_process(COMMAND git --git-dir=${CUB_SOURCE_DIR}/.git
                          --work-tree=${CUB_SOURCE_DIR} checkout ${CUB_TAG})
  # cub 1.16.0 is not compatible with current thrust version
  add_definitions(-DTHRUST_IGNORE_CUB_VERSION_CHECK)
else()
  set(CUB_TAG 1.8.0)
endif()

set(CUB_INCLUDE_DIR ${CUB_SOURCE_DIR})
message("CUB_INCLUDE_DIR is ${CUB_INCLUDE_DIR}")
include_directories(${CUB_INCLUDE_DIR})

ExternalProject_Add(
  extern_cub
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${CUB_SOURCE_DIR}
  PREFIX ${CUB_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(cub INTERFACE)

add_dependencies(cub extern_cub)
