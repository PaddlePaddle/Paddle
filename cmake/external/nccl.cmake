# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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

if(NOT WITH_GPU)
  return()
endif()

include(ExternalProject)

set(NCCL_SOURCE_DIR ${THIRD_PARTY_PATH}/nccl)

include_directories(${NCCL_SOURCE_DIR}/src/extern_nccl/src)

if(WITH_DSO)
  # If we use DSO, we do not build nccl, just download the dependencies
  set(NCCL_BUILD_COMMAND "")
  set(NCCL_INSTALL_COMMAND "")
  set(NCCL_INSTALL_DIR "")
else()
  # otherwise, we build nccl and link it.
  set(NCCL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/nccl)
  # Note: cuda 8.0 is needed to make nccl
  # When cuda is not installed on the system directory, need to set CUDA_HOME to your cuda root
  set(NCCL_BUILD_COMMAND "make -j 8")
  set(NCCL_INSTALL_COMMAND  "make install PREFIX=${NCCL_INSTALL_DIR}")
endif()

ExternalProject_Add(
    extern_nccl
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/NVIDIA/nccl.git"
    GIT_TAG         "v1.3.4-1"
    PREFIX          "${NCCL_SOURCE_DIR}"
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     "${NCCL_BUILD_COMMAND}"
    INSTALL_COMMAND   "${NCCL_INSTALL_COMMAND}"
    INSTALL_DIR       "${NCCL_INSTALL_DIR}"
    TEST_COMMAND      ""
)

if(WITH_DSO)
  if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/lib_nccl_dummy.c)
    file(WRITE ${dummyfile} "const char * dummy_nccl = \"${dummyfile}\";")
    add_library(nccl STATIC ${dummyfile})
  else()
    add_library(nccl INTERFACE)
  endif()
else()
  add_library(nccl STATIC IMPORTED GLOBAL)
  set_property(TARGET nccl PROPERTY IMPORTED_LOCATION
               ${NCCL_INSTALL_DIR}/lib/libnccl_static.a)
endif()

add_dependencies(nccl extern_nccl)
