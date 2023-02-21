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

set(K2_PATH "${THIRD_PARTY_PATH}/k2"
    CACHE STRING "A path setting for external_k2 path.")
set(K2_PREFIX_DIR ${K2_PATH})
set(K2_INSTALL_DIR ${THIRD_PARTY_PATH}/install/k2
    CACHE STRING "A path setting for external_k2 install path.")

# set(K2_REPOSITORY https://github.com/k2-fsa/k2.git)
# set(K2_TAG v1.23.4)

set(K2_REPOSITORY https://github.com/zh794390558/k2.git)
set(K2_TAG wo_pytorch)

ExternalProject_Add(
  extern_k2
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${K2_REPOSITORY}
  GIT_TAG ${K2_TAG}
  PREFIX ${K2_PREFIX_DIR}
  # UPDATE_COMMAND ""
  # CONFIGURE_COMMAND ""
  # BUILD_COMMAND ""
  # INSTALL_COMMAND ""
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX=${K2_INSTALL_DIR}
    -DBUILD_SHARED_LIBS=ON
    -DK2_USE_PYTORCH=OFF
    -DK2_ENABLE_TESTS=OFF
    -DK2_ENABLE_BENCHMARK=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
    ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${K2_INSTALL_DIR}
  BUILD_BYPRODUCTS ${K2_LIBRARIES}
  # TEST_COMMAND ""
  )


set(K2_INCLUDE_DIR ${K2_PREFIX_DIR}/src)
message("K2_INCLUDE_DIR is ${K2_INCLUDE_DIR}")
include_directories(${K2_INCLUDE_DIR})

add_library(k2 INTERFACE)

add_dependencies(k2 extern_k2)
