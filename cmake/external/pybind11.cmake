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
set(PYBIND_SOURCE_DIR ${PYBIND_PREFIX_DIR}/src/extern_pybind)
set(PYBIND_INCLUDE_DIR ${PYBIND_SOURCE_DIR}/include)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/pybind)
set(SOURCE_INCLUDE_DIR ${SOURCE_DIR}/include)

include_directories(${PYBIND_INCLUDE_DIR})

# It can be safely removed in gcc9.1+
set(PYBIND_PATCH_COMMAND "")
if(LINUX
   AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  set(PYBIND_TAG v2.13.6)
  file(TO_NATIVE_PATH
       ${PADDLE_SOURCE_DIR}/patches/pybind/detail/internals.h.patch native_dst)
  # Note: [Why calling some `git` commands before `patch`?]
  # Paddle's CI uses cache to accelerate the make process. However, error might raise when patch codes in two scenarios:
  # 1. Patch to the wrong version: the tag version of CI's cache falls behind PYBIND_TAG, use `git checkout ${PYBIND_TAG}` to solve this.
  # 2. Patch twice: the tag version of cache == PYBIND_TAG, but patch has already applied to cache.
  set(PYBIND_PATCH_COMMAND
      git checkout -- . && git checkout ${PYBIND_TAG} && patch -Nd
      ${SOURCE_INCLUDE_DIR}/pybind11/detail < ${native_dst})
endif()

ExternalProject_Add(
  extern_pybind
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX ${PYBIND_PREFIX_DIR}
  # If we explicitly leave the `UPDATE_COMMAND` of the ExternalProject_Add
  # function in CMakeLists blank, it will cause another parameter GIT_TAG
  # to be modified without triggering incremental compilation, and the
  # third-party library version changes cannot be incorporated.
  # reference: https://cmake.org/cmake/help/latest/module/ExternalProject.html
  UPDATE_COMMAND ""
  PATCH_COMMAND ${PYBIND_PATCH_COMMAND}
  CONFIGURE_COMMAND ""
  # I intentionally preserved an extern_pybind/include/pybind11 directory
  # to site-packages, so that you could discern that you intended to
  # employ not only python, but also CPP and were ready to incorporate header files.
  BUILD_COMMAND
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYBIND_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYBIND_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${SOURCE_INCLUDE_DIR}
          ${PYBIND_INCLUDE_DIR}
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(pybind INTERFACE)

add_dependencies(pybind extern_pybind)
