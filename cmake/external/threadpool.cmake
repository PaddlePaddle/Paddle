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

set(THREADPOOL_PREFIX_DIR ${THIRD_PARTY_PATH}/threadpool)
if(WITH_ASCEND OR WITH_ASCEND_CL)
  set(THREADPOOL_REPOSITORY https://gitee.com/tianjianhe/ThreadPool.git)
else()
  set(THREADPOOL_REPOSITORY ${GIT_URL}/progschj/ThreadPool.git)
endif()
set(THREADPOOL_TAG 9a42ec1329f259a5f4881a291db1dcb8f2ad9040)

set(THREADPOOL_INCLUDE_DIR ${THIRD_PARTY_PATH}/threadpool/src/extern_threadpool)
include_directories(${THREADPOOL_INCLUDE_DIR})

ExternalProject_Add(
  extern_threadpool
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${THREADPOOL_REPOSITORY}
  GIT_TAG ${THREADPOOL_TAG}
  PREFIX ${THREADPOOL_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(simple_threadpool INTERFACE)

add_dependencies(simple_threadpool extern_threadpool)
