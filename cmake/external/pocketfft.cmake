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

set(POCKETFFT_PATH
    "${THIRD_PARTY_PATH}/pocketfft"
    CACHE STRING "A path setting for external_pocketfft path.")
set(POCKETFFT_PREFIX_DIR ${POCKETFFT_PATH})

set(POCKETFFT_REPOSITORY https://gitlab.mpcdf.mpg.de/mtr/pocketfft.git)
set(POCKETFFT_TAG release_for_eigen)

set(POCKETFFT_INCLUDE_DIR ${POCKETFFT_PREFIX_DIR}/src)
message("POCKETFFT_INCLUDE_DIR is ${POCKETFFT_INCLUDE_DIR}")
include_directories(${POCKETFFT_INCLUDE_DIR})

ExternalProject_Add(
  extern_pocketfft
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${POCKETFFT_REPOSITORY}
  GIT_TAG ${POCKETFFT_TAG}
  PREFIX ${POCKETFFT_PREFIX_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(pocketfft INTERFACE)

add_dependencies(pocketfft extern_pocketfft)
