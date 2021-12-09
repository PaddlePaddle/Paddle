# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# Note(chenxin33): dirent.h is only exist in Linux, so get it from github when build in windows.
# use dirent tag v1.23.2 on 09/05//2018 https://github.com/tronkko/dirent.git

INCLUDE (ExternalProject)

SET(DIRENT_PREFIX_DIR       ${THIRD_PARTY_PATH}/dirent)
SET(DIRENT_INCLUDE_DIR      ${THIRD_PARTY_PATH}/dirent/src/extern_dirent/include)

include_directories(${DIRENT_INCLUDE_DIR})

set(DIRENT_REPOSITORY  ${GIT_URL}/tronkko/dirent)
set(DIRENT_TAG         1.23.2)

ExternalProject_Add(
  extern_dirent
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  GIT_REPOSITORY    ${DIRENT_REPOSITORY}
  GIT_TAG           ${DIRENT_TAG}
  PREFIX            ${DIRENT_PREFIX_DIR}
  UPDATE_COMMAND    ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

add_library(dirent INTERFACE)

add_dependencies(dirent extern_dirent)