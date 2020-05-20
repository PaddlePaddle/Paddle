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

set(CUB_PREFIX_DIR ${THIRD_PARTY_PATH}/cub)
set(CUB_SOURCE_DIR ${THIRD_PARTY_PATH}/cub/src/extern_cub)
set(CUB_REPOSITORY https://github.com/NVlabs/cub.git)
set(CUB_TAG        1.8.0)

cache_third_party(extern_cub
    REPOSITORY    ${CUB_REPOSITORY}
    TAG           ${CUB_TAG}
    DIR           CUB_SOURCE_DIR)

SET(CUB_INCLUDE_DIR   ${CUB_SOURCE_DIR})
include_directories(${CUB_INCLUDE_DIR})

ExternalProject_Add(
  extern_cub
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${CUB_DOWNLOAD_CMD}"
  PREFIX          ${CUB_PREFIX_DIR}
  SOURCE_DIR      ${CUB_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/cub_dummy.c)
  file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
  add_library(cub STATIC ${dummyfile})
else()
  add_library(cub INTERFACE)
endif()

add_dependencies(cub extern_cub)
