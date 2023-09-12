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

if(WIN32)
  message(SEND_ERROR "The current dgc support linux only")
  return()
endif()

include(ExternalProject)

set(DGC_DOWNLOAD_DIR ${PADDLE_SOURCE_DIR}/third_party/dgc/${CMAKE_SYSTEM_NAME})
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
set(DGC_CACHE_FILENAME "collective_7369ff.tgz")
set(DGC_URL_MD5 ede459281a0f979da8d84f81287369ff)

function(download_dgc)
  message(
    STATUS "Downloading ${DGC_URL} to ${DGC_DOWNLOAD_DIR}/${DGC_CACHE_FILENAME}"
  )
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(
    DOWNLOAD ${DGC_URL} ${DGC_DOWNLOAD_DIR}/${DGC_CACHE_FILENAME}
    EXPECTED_MD5 ${DGC_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${DGC_CACHE_FILENAME} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${DGC_CACHE_FILENAME} again"
    )
  endif()
endfunction()

if(EXISTS ${DGC_DOWNLOAD_DIR}/${DGC_CACHE_FILENAME})
  file(MD5 ${DGC_DOWNLOAD_DIR}/${DGC_CACHE_FILENAME} DGC_MD5)
  if(NOT DGC_MD5 STREQUAL DGC_URL_MD5)
    download_dgc()
  endif()
else()
  download_dgc()
endif()

ExternalProject_Add(
  extern_dgc
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${DGC_DOWNLOAD_DIR}/${DGC_CACHE_FILENAME}
  URL_MD5 ${DGC_URL_MD5}
  PREFIX "${DGC_PREFIX_DIR}"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make -j${NPROC}
  DOWNLOAD_DIR ${DGC_DOWNLOAD_DIR}
  SOURCE_DIR ${DGC_SOURCES_DIR}
  INSTALL_COMMAND
    mkdir -p ${DGC_INSTALL_DIR}/lib/ ${DGC_INCLUDE_DIR}/dgc && cp
    ${DGC_SOURCES_DIR}/build/lib/libdgc.a ${DGC_LIBRARIES} && cp
    ${DGC_SOURCES_DIR}/build/include/dgc.h ${DGC_INCLUDE_DIR}/dgc/
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS ${DGC_LIBRARIES})

add_library(dgc STATIC IMPORTED GLOBAL)
set_property(TARGET dgc PROPERTY IMPORTED_LOCATION ${DGC_LIBRARIES})
add_dependencies(dgc extern_dgc)
