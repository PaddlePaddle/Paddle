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

set(PYBIND_PREFIX_DIR ${THIRD_PARTY_PATH}/pybind)
set(PYBIND_REPOSITORY ${GIT_URL}/pybind/pybind11.git)
set(PYBIND_TAG v2.10.3)

set(PYBIND_INCLUDE_DIR ${THIRD_PARTY_PATH}/pybind/src/extern_pybind/include)
include_directories(${PYBIND_INCLUDE_DIR})
set(PYBIND_PATCH_COMMAND "")
if(NOT WIN32)
  file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/pybind/cast.h.patch
       native_dst)
  # Note: [Why calling some `git` commands before `patch`?]
  # Paddle's CI uses cache to accelarate the make process. However, error might raise when patch codes in two scenarios:
  # 1. Patch to the wrong version: the tag version of CI's cache falls behind PYBIND_TAG, use `git checkout ${PYBIND_TAG}` to solve this.
  # 2. Patch twice: the tag version of cache == PYBIND_TAG, but patch has already applied to cache.
  set(PYBIND_PATCH_COMMAND
      git checkout -- . && git checkout ${PYBIND_TAG} && patch -Nd
      ${PYBIND_INCLUDE_DIR}/pybind11 < ${native_dst})
endif()

ExternalProject_Add(
  extern_pybind
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${PYBIND_REPOSITORY}
  GIT_TAG ${PYBIND_TAG}
  PREFIX ${PYBIND_PREFIX_DIR}
  # If we explicitly leave the `UPDATE_COMMAND` of the ExternalProject_Add
  # function in CMakeLists blank, it will cause another parameter GIT_TAG
  # to be modified without triggering incremental compilation, and the
  # third-party library version changes cannot be incorporated.
  # reference: https://cmake.org/cmake/help/latest/module/ExternalProject.html
  UPDATE_COMMAND ""
  PATCH_COMMAND ${PYBIND_PATCH_COMMAND}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(pybind INTERFACE)

add_dependencies(pybind extern_pybind)
