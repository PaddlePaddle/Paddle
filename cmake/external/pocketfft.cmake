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

set(POCKETFFT_INCLUDE_DIR ${POCKETFFT_PREFIX_DIR}/src)
set(POCKETFFT_SOURCE_DIR ${POCKETFFT_PREFIX_DIR}/src/extern_pocketfft)
message("POCKETFFT_INCLUDE_DIR is ${POCKETFFT_INCLUDE_DIR}")
include_directories(${POCKETFFT_INCLUDE_DIR})

set(POCKETFFT_TAG release_for_eigen)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/pocketfft)

if(APPLE)
  file(TO_NATIVE_PATH
       ${PADDLE_SOURCE_DIR}/patches/pocketfft/pocketfft_hdronly.h.patch
       native_dst)
  set(POCKETFFT_PATCH_COMMAND
      git checkout -- . && git checkout ${POCKETFFT_TAG} && patch -Nd
      ${SOURCE_DIR} < ${native_dst})
endif()

ExternalProject_Add(
  extern_pocketfft
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${POCKETFFT_PREFIX_DIR}
  PATCH_COMMAND ${POCKETFFT_PATCH_COMMAND}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${SOURCE_DIR}
          ${POCKETFFT_SOURCE_DIR}
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(pocketfft INTERFACE)

add_dependencies(pocketfft extern_pocketfft)
