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
set(THREADPOOL_TAG 9a42ec1329f259a5f4881a291db1dcb8f2ad9040)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/threadpool)
set(THREADPOOL_INCLUDE_DIR ${SOURCE_DIR})
include_directories(${THREADPOOL_INCLUDE_DIR})

ExternalProject_Add(
  extern_threadpool
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${THREADPOOL_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(simple_threadpool INTERFACE)

add_dependencies(simple_threadpool extern_threadpool)
