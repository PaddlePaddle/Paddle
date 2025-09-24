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

set(GDRCOPY_HOME
    $ENV{GDRCOPY_HOME}
    CACHE PATH "Path to GDRCOPY installation")
if(GDRCOPY_HOME)
  message(STATUS "GDRCOPY_HOME: ${GDRCOPY_HOME}")
else()
  message(
    WARNING
      "Setting GDRCOPY_HOME environment or cmake option maybe needed to specify your install path GDRCOPY."
  )
endif()

set(NVSHMEM_INSTALL_DIR ${THIRD_PARTY_PATH}/install/nvshmem)
set(NVSHMEM_PREFIX_DIR ${THIRD_PARTY_PATH}/nvshmem)
set(NVSHMEM_SOURCE_DIR ${NVSHMEM_PREFIX_DIR}/src/extern_nvshmem)
message(STATUS "NVSHMEM_INSTALL_DIR: ${NVSHMEM_INSTALL_DIR}")

set(NVSHMEM_INCLUDE_DIR
    "${NVSHMEM_INSTALL_DIR}/include"
    CACHE PATH "nvshmem include directory." FORCE)

include_directories(${NVSHMEM_INCLUDE_DIR})

set(NVSHMEM_TAR_NAME "nvshmem_src_3.2.5-1.txz")

if(NVSHMEM_SRC_TAR_PATH)
  set(NVSHMEM_DOWNLOAD_COMMAND
      rm -rf extern_nvshmem ${NVSHMEM_TAR_NAME} && cp ${NVSHMEM_SRC_TAR_PATH} .
      && tar xf ${NVSHMEM_TAR_NAME} && mv nvshmem_src extern_nvshmem)
else()
  set(NVSHMEM_URL
      "https://paddle-ci.gz.bcebos.com/${NVSHMEM_TAR_NAME}"
      CACHE STRING "" FORCE)
  set(NVSHMEM_DOWNLOAD_COMMAND
      rm -rf extern_nvshmem ${NVSHMEM_TAR_NAME} && wget --no-check-certificate
      -q ${NVSHMEM_URL} && tar xf ${NVSHMEM_TAR_NAME} && mv nvshmem_src
      extern_nvshmem)
endif()

set(NVSHMEM_PATCH_PATH ${PADDLE_SOURCE_DIR}/patches/nvshmem/nvshmem.patch)
set(NVSHMEM_PATCH_COMMAND
    git init && git config --global --add safe.directory ${NVSHMEM_SOURCE_DIR}
    && git config user.name "PaddlePaddle" && git config user.email
    "paddle@baidu.com" && git add . && git commit -m "init" && git apply
    ${NVSHMEM_PATCH_PATH})

set(NVSHMEM_LIB ${NVSHMEM_INSTALL_DIR}/lib/libnvshmem.a)
set(NVSHMEM_BOOTSTRAP_UID_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_uid.so.3)
set(NVSHMEM_BOOTSTRAP_PMI_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_pmi.so.3)
set(NVSHMEM_BOOTSTRAP_PMI2_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_pmi2.so.3)
set(NVSHMEM_TRANSPORT_IBRC_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_transport_ibrc.so.3)
set(NVSHMEM_TRANSPORT_IBGDA_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_transport_ibgda.so.3)

# only compile nvshmem for sm90
set(CUDA_ARCHITECTURES "90")

ExternalProject_Add(
  extern_nvshmem
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  PREFIX ${NVSHMEM_PREFIX_DIR}
  SOURCE_DIR ${NVSHMEM_SOURCE_DIR}
  DOWNLOAD_DIR ${NVSHMEM_PREFIX_DIR}/src
  DOWNLOAD_COMMAND ${NVSHMEM_DOWNLOAD_COMMAND}
  PATCH_COMMAND ${NVSHMEM_PATCH_COMMAND}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${NVSHMEM_INSTALL_DIR}
             -DGDRCOPY_HOME:PATH=${GDRCOPY_HOME}
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
             -DNVSHMEM_ENABLE_ALL_DEVICE_INLINING=0
             -DNVSHMEM_SHMEM_SUPPORT=0
             -DNVSHMEM_UCX_SUPPORT=0
             -DNVSHMEM_USE_NCCL=0
             -DNVSHMEM_IBGDA_SUPPORT=1
             -DNVSHMEM_PMIX_SUPPORT=0
             -DNVSHMEM_TIMEOUT_DEVICE_POLLING=0
             -DNVSHMEM_USE_GDRCOPY=1
             -DNVSHMEM_IBRC_SUPPORT=1
             -DNVSHMEM_BUILD_TESTS=0
             -DNVSHMEM_BUILD_EXAMPLES=0
             -DNVSHMEM_MPI_SUPPORT=0
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${NVSHMEM_INSTALL_DIR}
  BUILD_BYPRODUCTS ${NVSHMEM_LIB})

add_definitions(-DPADDLE_WITH_NVSHMEM)
add_library(nvshmem STATIC IMPORTED GLOBAL)
set_property(TARGET nvshmem PROPERTY IMPORTED_LOCATION ${NVSHMEM_LIB})
add_dependencies(nvshmem extern_nvshmem)
