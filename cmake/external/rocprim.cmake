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

# rocprim is "ROCm Parallel Primitives" for short.
# It is a header-only library providing HIP and HC parallel primitives
# for developing performant GPU-accelerated code on AMD ROCm platform.

if("x${HCC_HOME}" STREQUAL "x")
  set(HCC_HOME "/opt/rocm/hcc")
endif()

INCLUDE(ExternalProject)

SET(ROCPRIM_SOURCE_DIR ${THIRD_PARTY_PATH}/rocprim)
SET(ROCPRIM_INSTALL_DIR  ${THIRD_PARTY_PATH}/install/rocprim)
SET(ROCPRIM_INCLUDE_DIR ${ROCPRIM_INSTALL_DIR}/include)

ExternalProject_Add(
    extern_rocprim
    ${SHALLOW_CLONE}
    GIT_REPOSITORY "https://github.com/ROCmSoftwarePlatform/rocPRIM.git"
    GIT_TAG        5bd41b96ab8d8343330fb2c3e1b96775bde3b3fc 
    PREFIX         ${ROCPRIM_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS     -DCMAKE_CXX_COMPILER=${HCC_HOME}/bin/hcc
    CMAKE_ARGS     -DONLY_INSTALL=ON
    CMAKE_ARGS     -DBUILD_TEST=OFF
    CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=${ROCPRIM_INSTALL_DIR}

    INSTALL_DIR    ${ROCPRIM_INSTALL_DIR}
    ${EXTERNAL_PROJECT_LOG_ARGS}
)

INCLUDE_DIRECTORIES(${ROCPRIM_INCLUDE_DIR})

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/rocprim_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_rocprim = \"${dummyfile}\";")
    add_library(rocprim STATIC ${dummyfile})
else()
    add_library(rocprim INTERFACE)
endif()

add_dependencies(rocprim extern_rocprim)
