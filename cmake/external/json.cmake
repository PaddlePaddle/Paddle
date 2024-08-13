# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

set(JSON_PREFIX_DIR ${THIRD_PARTY_PATH}/nlohmann_json)
set(JSON_INCLUDE_DIR ${JSON_PREFIX_DIR}/include)

set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/nlohmann_json)
set(SOURCE_INCLUDE_DIR ${SOURCE_DIR}/include)

include_directories(${SOURCE_INCLUDE_DIR})

set(JSON_BuildTests
    OFF
    CACHE INTERNAL "")

ExternalProject_Add(
  extern_json
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${JSON_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_IN_SOURCE 1
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(json INTERFACE)
#target_include_directories(json PRIVATE ${JSON_INCLUDE_DIR})
add_dependencies(json extern_json)
