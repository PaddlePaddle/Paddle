# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# Note(chenxin33): dirent.h is only exist in Linux, so get it from github when build in windows.
# use dirent tag v1.23.2 on 09/05//2018 https://github.com/tronkko/dirent.git

include(ExternalProject)

if((NOT DEFINED DIRENT_NAME) OR (NOT DEFINED DIRENT_URL))
  set(DIRENT_VER
      "1.23.2"
      CACHE STRING "" FORCE)
  set(DIRENT_NAME
      "dirent"
      CACHE STRING "" FORCE)
  set(DIRENT_URL
      "https://github.com/tronkko/dirent/archive/refs/tags/1.23.2.tar.gz"
      CACHE STRING "" FORCE)
  set(DIRENT_CACHE_FILENAME "1.23.2.tar.gz")
endif()

message(STATUS "DIRENT_NAME: ${DIRENT_NAME}, DIRENT_URL: ${DIRENT_URL}")
set(DIRENT_DOWNLOAD_DIR "${PADDLE_SOURCE_DIR}/third_party/dirent")
set(DIRENT_PREFIX_DIR ${THIRD_PARTY_PATH}/dirent)
set(DIRENT_INCLUDE_DIR ${THIRD_PARTY_PATH}/dirent/src/extern_dirent/include)

include_directories(${DIRENT_INCLUDE_DIR})

function(download_dirent)
  message(
    STATUS
      "Downloading ${DIRENT_URL} to ${DIRENT_DOWNLOAD_DIR}/${DIRENT_CACHE_FILENAME}"
  )
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(DOWNLOAD ${DIRENT_URL} ${DIRENT_DOWNLOAD_DIR}/${DIRENT_CACHE_FILENAME}
       STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${DIRENT_CACHE_FILENAME} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${DIRENT_CACHE_FILENAME} again"
    )
  endif()
endfunction()

if(NOT EXISTS ${DIRENT_DOWNLOAD_DIR}/${DIRENT_CACHE_FILENAME})
  message(
    STATUS
      "Local package ${DIRENT_CACHE_FILENAME} not found, starting download.")
  download_dirent()
endif()

ExternalProject_Add(
  extern_dirent
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  URL ${DIRENT_DOWNLOAD_DIR}/${DIRENT_CACHE_FILENAME}
  PREFIX ${DIRENT_PREFIX_DIR}
  DOWNLOAD_DIR ${DIRENT_DOWNLOAD_DIR}
  SOURCE_DIR ${DIRENT_INSTALL_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(dirent INTERFACE)

add_dependencies(dirent extern_dirent)
