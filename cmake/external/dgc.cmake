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

set(DGC_PREFIX_DIR "${THIRD_PARTY_PATH}/dgc")
set(DGC_SOURCES_DIR "${THIRD_PARTY_PATH}/dgc/src/extern_dgc")
set(DGC_INSTALL_DIR "${THIRD_PARTY_PATH}/install/dgc")
set(DGC_INCLUDE_DIR
    "${DGC_INSTALL_DIR}/include"
    CACHE PATH "dgc include directory." FORCE)
set(DGC_LIBRARIES
    "${DGC_INSTALL_DIR}/lib/libdgc.a"
    CACHE FILEPATH "dgc library." FORCE)
set(DGC_URL "https://fleet.bj.bcebos.com/dgc/collective_7369ff.tgz")
include_directories(${DGC_INCLUDE_DIR})

ExternalProject_Add(
  extern_dgc
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${DGC_URL}
  URL_MD5 "ede459281a0f979da8d84f81287369ff"
  PREFIX "${DGC_PREFIX_DIR}"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make -j${NPROC}
  INSTALL_COMMAND
    mkdir -p ${DGC_INSTALL_DIR}/lib/ ${DGC_INCLUDE_DIR}/dgc && cp
    ${DGC_SOURCES_DIR}/build/lib/libdgc.a ${DGC_LIBRARIES} && cp
    ${DGC_SOURCES_DIR}/build/include/dgc.h ${DGC_INCLUDE_DIR}/dgc/
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS ${DGC_LIBRARIES})

add_library(dgc STATIC IMPORTED GLOBAL)
set_property(TARGET dgc PROPERTY IMPORTED_LOCATION ${DGC_LIBRARIES})
add_dependencies(dgc extern_dgc)
