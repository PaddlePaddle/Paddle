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

if(NOT WITH_CUCOLLECTIONS)
  return()
endif()

if(WITH_ARM OR WIN32)
  message(SEND_ERROR "The current cuCollections support linux only")
  return()
endif()

include(ExternalProject)

set(CUCOLLECTIONS_PREFIX_DIR ${THIRD_PARTY_PATH}/cuCollections)

set(CUCOLLECTIONS_REPOSITORY https://github.com/NVIDIA/cuCollections.git)
set(CUCOLLECTIONS_TAG pair-atomic)

include_directories(
  "${THIRD_PARTY_PATH}/cuCollections/src/extern_cuCollections/include/")

ExternalProject_Add(
  extern_cuCollections
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${CUCOLLECTIONS_REPOSITORY}
  GIT_TAG "${CUCOLLECTIONS_TAG}"
  PREFIX ${CUCOLLECTIONS_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(cuCollections INTERFACE)

add_dependencies(cuCollections extern_cuCollections)
