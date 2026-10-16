# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
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

# Eigen 3.4.1 is pinned by the third_party/eigen3 gitlink.
set(EIGEN_PREFIX_DIR ${THIRD_PARTY_PATH}/eigen3)
set(EIGEN_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/eigen3)

if(WIN32)
  add_definitions(-DEIGEN_STRONG_INLINE=inline)
endif()

# Eigen 3.4.1 uses std::ssize without including <iterator>; TUs that don't
# transitively include it (e.g. the DCU toolchain) fail to compile. Apply the
# upstream feature-test guard to a build-dir copy of the headers so the
# submodule working tree stays pristine (CI switches branches in the same
# tree for API diff checks). Drop this when Eigen is upgraded past 3.4.1.
set(EIGEN_PATCHED_DIR ${EIGEN_PREFIX_DIR}/patched_src)
file(COPY ${EIGEN_SOURCE_DIR}/Eigen DESTINATION ${EIGEN_PATCHED_DIR})
file(COPY ${EIGEN_SOURCE_DIR}/unsupported DESTINATION ${EIGEN_PATCHED_DIR})
set(_eigen_meta_h ${EIGEN_PATCHED_DIR}/Eigen/src/Core/util/Meta.h)
file(READ ${_eigen_meta_h} _eigen_meta_h_content)
set(_eigen_meta_h_anchor "#if EIGEN_COMP_CXXVER < 20\n")
set(_eigen_meta_h_guard
    "#if EIGEN_COMP_CXXVER < 20 || !defined(__cpp_lib_ssize) || \\\n    __cpp_lib_ssize < 201902L\n"
)
string(FIND "${_eigen_meta_h_content}" "${_eigen_meta_h_anchor}"
            _eigen_meta_h_anchor_pos)
if(_eigen_meta_h_anchor_pos EQUAL -1)
  message(
    FATAL_ERROR
      "Anchor '#if EIGEN_COMP_CXXVER < 20' not found in Eigen Meta.h; "
      "the std::ssize guard fix may already be upstream, drop this workaround.")
endif()
string(REPLACE "${_eigen_meta_h_anchor}" "${_eigen_meta_h_guard}"
               _eigen_meta_h_content "${_eigen_meta_h_content}")
file(WRITE ${_eigen_meta_h} "${_eigen_meta_h_content}")

set(EIGEN_INCLUDE_DIR ${EIGEN_PATCHED_DIR})
# Use SYSTEM include to suppress warnings from Eigen third-party headers.
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})
ExternalProject_Add(
  extern_eigen3
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${EIGEN_SOURCE_DIR}
  PREFIX ${EIGEN_PREFIX_DIR}
  CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(eigen3 INTERFACE)

add_dependencies(eigen3 extern_eigen3)

# sw not support thread_local semantic
if(WITH_SW)
  add_definitions(-DEIGEN_AVOID_THREAD_LOCAL)
endif()
