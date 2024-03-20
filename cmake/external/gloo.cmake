# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

set(GLOO_PROJECT "extern_gloo")
set(GLOO_PREFIX_DIR ${THIRD_PARTY_PATH}/gloo)
set(GLOO_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/gloo)
set(GLOO_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gloo)
set(GLOO_INCLUDE_DIR
    ${GLOO_INSTALL_DIR}/include
    CACHE PATH "gloo include directory." FORCE)
set(GLOO_LIBRARY_DIR
    ${GLOO_INSTALL_DIR}/lib
    CACHE PATH "gloo library directory." FORCE)

# As we add extra features for gloo, we use the non-official repo
set(GLOO_TAG v0.0.3)
set(GLOO_LIBRARIES
    ${GLOO_INSTALL_DIR}/lib/libgloo.a
    CACHE FILEPATH "gloo library." FORCE)

# Setup gloo patch command
set(GLOO_PATCH_COMMAND git checkout -- . && git checkout ${GLOO_TAG})

file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/gloo/device.cc.patch
     native_dst)
file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/gloo/types.h.patch
     types_header)
file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/gloo/linux.cc.patch
     linux_cc_ethtool)

# cmake-format: off
list(APPEND GLOO_PATCH_COMMAND
    && patch -Nd ${GLOO_SOURCE_DIR}/gloo/transport/tcp < ${native_dst}
    && patch -Nd ${GLOO_SOURCE_DIR}/gloo/ < ${types_header}
    && patch -Nd ${GLOO_SOURCE_DIR}/gloo/common/ < ${linux_cc_ethtool})
# cmake-format: on

set(GLOO_CMAKE_C_FLAGS "-O3 -fPIC")
set(GLOO_CMAKE_CXX_FLAGS "-O3 -fPIC")

ExternalProject_Add(
  ${GLOO_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${GLOO_SOURCE_DIR}
  PREFIX ${GLOO_PREFIX_DIR}
  PATCH_COMMAND ${GLOO_PATCH_COMMAND}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
             -DCMAKE_INSTALL_PREFIX=${GLOO_INSTALL_DIR}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_FLAGS=${GLOO_CMAKE_C_FLAGS}
             -DCMAKE_CXX_FLAGS=${GLOO_CMAKE_CXX_FLAGS}
  BUILD_BYPRODUCTS ${GLOO_LIBRARIES})

add_library(gloo STATIC IMPORTED GLOBAL)
set_property(TARGET gloo PROPERTY IMPORTED_LOCATION ${GLOO_LIBRARIES})
add_dependencies(gloo ${GLOO_PROJECT})

include_directories(${GLOO_INCLUDE_DIR})
