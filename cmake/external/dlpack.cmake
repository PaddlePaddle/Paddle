# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

set(DLPACK_PREFIX_DIR ${THIRD_PARTY_PATH}/dlpack)
set(DLPACK_TAG v0.8)
set(DLPACK_INCLUDE_DIR ${THIRD_PARTY_PATH}/dlpack/src/extern_dlpack/include)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/dlpack)
include_directories(${SOURCE_DIR}/include)

ExternalProject_Add(
  extern_dlpack
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${DLPACK_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(dlpack INTERFACE)

add_dependencies(dlpack extern_dlpack)
