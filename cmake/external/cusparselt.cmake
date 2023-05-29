# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT (WITH_CUSPARSELT AND WITH_TENSORRT))
  return()
endif()

if(WITH_ARM OR WIN32)
  message(SEND_ERROR "The current sparselt support linux only")
  return()
endif()

include(ExternalProject)

set(CUSPARSELT_PROJECT "extern_cusparselt")
set(CUSPARSELT_P "https://developer.download.nvidia.com/compute")
set(CUSPARSELT_F "libcusparse_lt-linux-x86_64-0.2.0.1.tar.gz")
set(CUSPARSELT_URL
    "${CUSPARSELT_P}/libcusparse-lt/0.2.0/local_installers/${CUSPARSELT_F}"
    CACHE STRING "" FORCE)
set(CUSPARSELT_PREFIX_DIR ${THIRD_PARTY_PATH}/cusparselt)
set(CUSPARSELT_INSTALL_DIR ${THIRD_PARTY_PATH}/install/cusparselt)
set(CUSPARSELT_INC_DIR
    "${CUSPARSELT_INSTALL_DIR}/include"
    CACHE PATH "sparselt include directory." FORCE)
set(CUSPARSELT_LIB_DIR
    "${CUSPARSELT_INSTALL_DIR}/lib64"
    CACHE PATH "sparselt lib directory." FORCE)
set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)
include_directories(${CUSPARSELT_INC_DIR})

ExternalProject_Add(
  ${CUSPARSELT_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${CUSPARSELT_URL}
  PREFIX ${CUSPARSELT_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    ${CUSPARSELT_PREFIX_DIR}/src/extern_cusparselt/lib64 ${CUSPARSELT_LIB_DIR}
    && ${CMAKE_COMMAND} -E copy_directory
    ${CUSPARSELT_PREFIX_DIR}/src/extern_cusparselt/include ${CUSPARSELT_INC_DIR}
  UPDATE_COMMAND "")

add_library(cusparselt INTERFACE)
add_dependencies(cusparselt ${CUSPARSELT_PROJECT})
set(CUSPARSELT_FOUND ON)
add_definitions(-DPADDLE_WITH_CUSPARSELT)
