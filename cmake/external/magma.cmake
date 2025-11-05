# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

set(MAGMA_PREFIX_DIR ${THIRD_PARTY_PATH}/magma)
set(MAGMA_DOWNLOAD_DIR
    ${PADDLE_SOURCE_DIR}/third_party/magma/${CMAKE_SYSTEM_NAME})
set(MAGMA_INSTALL_DIR ${THIRD_PARTY_PATH}/install/magma)
set(MAGMA_LIB_DIR ${MAGMA_INSTALL_DIR}/lib)

# Note(zhouwei): magma need fortran compiler which many machines don't have, so use precompiled library.
# use magma tag v3.10.0 on 06/28/2021 https://github.com/Reference-MAGMA/magma
if(LINUX)
  set(MAGMA_FILE
      "magma_lnx_v2.9.0.20251105.tar.gz"
      CACHE STRING "" FORCE)
  set(MAGMA_URL
      "https://paddlepaddledeps.bj.bcebos.com/${MAGMA_FILE}"
      CACHE STRING "" FORCE)
  set(MAGMA_URL_MD5 80bf6529ee8aa5248a3c5a476f4ab5f2)
  set(MAGMA_LIB "${MAGMA_LIB_DIR}/libmagma.so")
elseif(WIN32)
  message(FATAL_ERROR "magma do not support windows")
  # Refer to [magma-for-windows] http://icl.cs.utk.edu/magma-for-windows/magma/#magmae
  set(MAGMA_FILE
      "magma_win_v3.10.0.20210628.zip"
      CACHE STRING "" FORCE)
  set(MAGMA_URL
      "https://paddlepaddledeps.bj.bcebos.com/${MAGMA_FILE}"
      CACHE STRING "" FORCE)
  set(MAGMA_URL_MD5 590d080392dcd5abbd5dca767a50b63a)
  set(MAGMA_LIB "${MAGMA_LIB_DIR}/libmagma.dll")
else() # MacOS
  message(FATAL_ERROR "magma do not support macos")
  if(APPLE AND WITH_ARM)
    set(MAGMA_FILE
        "magma_mac_arm64_v0.3.26.tar.gz"
        CACHE STRING "" FORCE)
    set(MAGMA_URL
        "https://paddlepaddledeps.bj.bcebos.com/${MAGMA_FILE}"
        CACHE STRING "" FORCE)
    set(MAGMA_URL_MD5 3f6412105ae2b7465e5ee90c8673e6d4)
    set(MAGMA_LIB "${MAGMA_LIB_DIR}/libmagma.dylib")
  else()
    set(MAGMA_FILE
        "magma_mac_v3.10.0.20210628.tar.gz"
        CACHE STRING "" FORCE)
    set(MAGMA_URL
        "https://paddlepaddledeps.bj.bcebos.com/${MAGMA_FILE}"
        CACHE STRING "" FORCE)
    set(MAGMA_URL_MD5 427aecf8dee8523de3566ca8e47944d7)
    set(MAGMA_LIB "${MAGMA_LIB_DIR}/libmagma.3.dylib")
  endif()
endif()

function(download_magma)
  message(
    STATUS "Downloading ${MAGMA_URL} to ${MAGMA_DOWNLOAD_DIR}/${MAGMA_FILE}")
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(
    DOWNLOAD ${MAGMA_URL} ${MAGMA_DOWNLOAD_DIR}/${MAGMA_FILE}
    EXPECTED_MD5 ${MAGMA_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${MAGMA_FILE} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${MAGMA_FILE} again"
    )
  endif()
endfunction()

# Download and check magma.
if(EXISTS ${MAGMA_DOWNLOAD_DIR}/${MAGMA_FILE})
  file(MD5 ${MAGMA_DOWNLOAD_DIR}/${MAGMA_FILE} MAGMA_MD5)
  if(NOT MAGMA_MD5 STREQUAL MAGMA_URL_MD5)
    # clean build file
    file(REMOVE_RECURSE ${MAGMA_PREFIX_DIR})
    file(REMOVE_RECURSE ${MAGMA_INSTALL_DIR})
    download_magma()
  endif()
else()
  download_magma()
endif()

ExternalProject_Add(
  extern_magma
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${MAGMA_DOWNLOAD_DIR}/${MAGMA_FILE}
  URL_MD5 ${MAGMA_URL_MD5}
  DOWNLOAD_DIR ${MAGMA_DOWNLOAD_DIR}
  SOURCE_DIR ${MAGMA_LIB_DIR}
  PREFIX ${MAGMA_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  PATCH_COMMAND ""
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${MAGMA_LIB})
