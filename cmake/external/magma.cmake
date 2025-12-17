# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# use magma tag v2.9.0 on 07/28/2025 https://github.com/icl-utk-edu/magma/tree/v2.9.0
if(LINUX)
  if(WITH_GPU)
    execute_process(
      COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
      OUTPUT_VARIABLE CUDA_ARCH_LIST
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Detected CUDA architectures: ${CUDA_ARCH_LIST}")

    set(MAGMA_ARCH "sm_70")
    set(MAGMA_URL_MD5 "649b886f9df75face6aa98015ccb1885")

    if(CUDA_ARCH_LIST MATCHES "compute_80")
      set(MAGMA_ARCH "sm_80")
      set(MAGMA_URL_MD5 "c16079b2eaf48f5af741d979c5090667")
    endif()

    execute_process(
      COMMAND nvcc -V
      OUTPUT_VARIABLE NVCC_VERSION_RAW
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" _ ${NVCC_VERSION_RAW})
    set(CUDA_MAJOR ${CMAKE_MATCH_1})
    set(CUDA_MINOR ${CMAKE_MATCH_2})
    if(CUDA_MAJOR EQUAL 12 AND CUDA_MINOR EQUAL 0)
      set(MAGMA_ARCH "cuda120")
      set(MAGMA_URL_MD5 "a00f89aeb81f0373aac34684695847ab")
    endif()
  endif()

  if(WITH_XPU)
    set(MAGMA_ARCH "xpu")
    set(MAGMA_URL_MD5 "889f990bfd4a24c98831cfda0243674a")
  endif()

  if(WITH_ROCM)
    set(MAGMA_ARCH "hip")
    set(MAGMA_URL_MD5 "f4fbe46c665819f6ae86d1aa447d07b5")
  endif()

  message(STATUS "Selected MAGMA architecture: ${MAGMA_ARCH}")
  message(STATUS "Selected MAGMA URL MD5: ${MAGMA_URL_MD5}")

  set(MAGMA_FILE
      "magma_lnx_${MAGMA_ARCH}_v2.9.0.20250728.tar.gz"
      CACHE STRING "" FORCE)
  set(MAGMA_URL
      "https://paddlepaddledeps.bj.bcebos.com/${MAGMA_FILE}"
      CACHE STRING "" FORCE)
  set(MAGMA_LIB "${MAGMA_LIB_DIR}/libmagma.so")
elseif(WIN32)
  message("magma do not support windows yet, skip ...")
else() # MacOS
  message("magma do not support macos or other platform yet, skip ...")
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

if(WITH_MAGMA)
  add_definitions(-DPADDLE_WITH_MAGMA)
endif()
