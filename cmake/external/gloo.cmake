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
set(GLOO_REPOSITORY ${GIT_URL}/ziyoujiyi/gloo.git)
set(GLOO_TAG v0.0.3)
set(GLOO_LIBRARIES
    "${GLOO_INSTALL_DIR}/lib/libgloo.a"
    CACHE FILEPATH "gloo library." FORCE)

include_directories(${GLOO_INCLUDE_DIR})

if(WITH_ASCEND OR WITH_ASCEND_CL)
  ExternalProject_Add(
    ${GLOO_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
    GIT_REPOSITORY ${GLOO_REPOSITORY}
    GIT_TAG ${GLOO_TAG}
    PREFIX "${GLOO_PREFIX_DIR}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
      mkdir -p ${GLOO_SOURCE_DIR}/build && cd ${GLOO_SOURCE_DIR}/build && cmake
      .. -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} && make && mkdir -p
      ${GLOO_LIBRARY_DIR} ${GLOO_INCLUDE_DIR}/gloo
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy
                    ${GLOO_SOURCE_DIR}/build/gloo/libgloo.a ${GLOO_LIBRARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${GLOO_SOURCE_DIR}/gloo/"
            "${GLOO_INCLUDE_DIR}/gloo"
    BUILD_BYPRODUCTS ${GLOO_LIBRARIES})
else()
  ExternalProject_Add(
    ${GLOO_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
    GIT_REPOSITORY ${GLOO_REPOSITORY}
    GIT_TAG ${GLOO_TAG}
    PREFIX "${GLOO_PREFIX_DIR}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
      mkdir -p ${GLOO_SOURCE_DIR}/build && cd ${GLOO_SOURCE_DIR}/build && cmake
      .. -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} && make && mkdir -p
      ${GLOO_LIBRARY_DIR} ${GLOO_INCLUDE_DIR}/gloo
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy
                    ${GLOO_SOURCE_DIR}/build/gloo/libgloo.a ${GLOO_LIBRARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${GLOO_SOURCE_DIR}/gloo/"
            "${GLOO_INCLUDE_DIR}/gloo"
    BUILD_BYPRODUCTS ${GLOO_LIBRARIES})
endif()

add_library(gloo STATIC IMPORTED GLOBAL)
set_property(TARGET gloo PROPERTY IMPORTED_LOCATION ${GLOO_LIBRARIES})
add_dependencies(gloo ${GLOO_PROJECT})
