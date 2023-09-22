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
set(GLOO_SOURCE_DIR ${THIRD_PARTY_PATH}/gloo/src/extern_gloo)
set(GLOO_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gloo)
set(GLOO_INCLUDE_DIR
    "${GLOO_INSTALL_DIR}/include"
    CACHE PATH "gloo include directory." FORCE)
set(GLOO_LIBRARY_DIR
    "${GLOO_INSTALL_DIR}/lib"
    CACHE PATH "gloo library directory." FORCE)
# As we add extra features for gloo, we use the non-official repo
set(GLOO_TAG v0.0.3)
set(GLOO_LIBRARIES
    "${GLOO_INSTALL_DIR}/lib/libgloo.a"
    CACHE FILEPATH "gloo library." FORCE)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/gloo)
set(GLOO_PATCH_COMMAND "")
if(WITH_GPU)
  if(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0 AND ${CMAKE_CXX_COMPILER_VERSION}
                                                  VERSION_GREATER 12.0)
    file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/gloo/device.cc.patch
         native_dst)
    set(GLOO_PATCH_COMMAND
        git checkout -- . && git checkout ${GLOO_TAG} && patch -Nd
        ${SOURCE_DIR}/gloo/transport/tcp < ${native_dst})
  endif()
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpfullversion -dumpversion
                  OUTPUT_VARIABLE GCC_VERSION)
  string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
  list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
  list(GET GCC_VERSION_COMPONENTS 1 GCC_MINOR)
  set(GCC_VERSION "${GCC_MAJOR}.${GCC_MINOR}")
  if(GCC_VERSION GREATER_EQUAL "12.0")
    file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/gloo/device.cc.patch
         native_dst)
    file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/gloo/types.h.patch
         types_header)
    # See: [Why calling some `git` commands before `patch`?]
    set(GLOO_PATCH_COMMAND
        git checkout -- . && git checkout ${GLOO_TAG} && patch -Nd
        ${SOURCE_DIR}/gloo/transport/tcp < ${native_dst} && patch -Nd
        ${SOURCE_DIR}/gloo/ < ${types_header})
  endif()
endif()
include_directories(${GLOO_INCLUDE_DIR})

ExternalProject_Add(
  ${GLOO_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  PREFIX "${GLOO_PREFIX_DIR}"
  UPDATE_COMMAND ""
  PATCH_COMMAND ${GLOO_PATCH_COMMAND}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
    mkdir -p ${GLOO_SOURCE_DIR}/build && cd ${GLOO_SOURCE_DIR}/build && cmake
    ${SOURCE_DIR} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} && ${CMAKE_COMMAND}
    --build . && mkdir -p ${GLOO_LIBRARY_DIR} ${GLOO_INCLUDE_DIR}/glo
  INSTALL_COMMAND ${CMAKE_COMMAND} -E copy
                  ${GLOO_SOURCE_DIR}/build/gloo/libgloo.a ${GLOO_LIBRARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${SOURCE_DIR}/gloo/"
          "${GLOO_INCLUDE_DIR}/gloo"
  BUILD_BYPRODUCTS ${GLOO_LIBRARIES})

add_library(gloo STATIC IMPORTED GLOBAL)
set_property(TARGET gloo PROPERTY IMPORTED_LOCATION ${GLOO_LIBRARIES})
add_dependencies(gloo ${GLOO_PROJECT})
