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

# Note(zhouwei): extern_cub  has code __FILE_, If the path of extern_cub is changed,
# it will effect about 30+ cu files sccache hit and slow compile speed  on windows.
# Therefore, a fixed CUB_PATH will be input to increase the sccache hit rate.
set(CUB_PATH
    "${THIRD_PARTY_PATH}/cub"
    CACHE STRING "A path setting for external_cub path.")
set(CUB_PREFIX_DIR ${CUB_PATH})

set(CUB_REPOSITORY ${GIT_URL}/NVlabs/cub.git)

if(WIN32 AND ${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL 11.6)
  # cuda_11.6.2_511.65‘s own cub is 1.15.0, which will cause compiling error in windows.
  set(CUB_TAG 1.16.0)
  # cub 1.16.0 is not compitable with current thrust version
  add_definitions(-DTHRUST_IGNORE_CUB_VERSION_CHECK)
else()
  set(CUB_TAG 1.8.0)
endif()

set(CUB_INCLUDE_DIR ${CUB_PREFIX_DIR}/src/extern_cub)
message("CUB_INCLUDE_DIR is ${CUB_INCLUDE_DIR}")
include_directories(${CUB_INCLUDE_DIR})

ExternalProject_Add(
  extern_cub
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${CUB_REPOSITORY}
  GIT_TAG ${CUB_TAG}
  PREFIX ${CUB_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(cub INTERFACE)

add_dependencies(cub extern_cub)
