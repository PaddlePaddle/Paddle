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

set(CUTLASS_PREFIX_DIR ${THIRD_PARTY_PATH}/cutlass)

set(CUTLASS_REPOSITORY https://github.com/NVIDIA/cutlass.git)
set(CUTLASS_TAG v2.10.0)

include_directories("${THIRD_PARTY_PATH}/cutlass/src/extern_cutlass/")
include_directories("${THIRD_PARTY_PATH}/cutlass/src/extern_cutlass/include/")
include_directories(
  "${THIRD_PARTY_PATH}/cutlass/src/extern_cutlass/tools/util/include/")

add_definitions("-DPADDLE_WITH_CUTLASS")

ExternalProject_Add(
  extern_cutlass
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${CUTLASS_REPOSITORY}
  GIT_TAG "${CUTLASS_TAG}"
  PREFIX ${CUTLASS_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(cutlass INTERFACE)

add_dependencies(cutlass extern_cutlass)
