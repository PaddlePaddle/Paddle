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
set(CUSPARSELT_DOWNLOAD_DIR
    ${PADDLE_SOURCE_DIR}/third_party/cusparselt/${CMAKE_SYSTEM_NAME})
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
set(CUSPARSELT_CACHE_FILENAME "${CUSPARSELT_F}")
set(CUSPARSELT_URL_MD5 "4f72f469e9cb1a85b09017fbace733d7")

set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)
include_directories(${CUSPARSELT_INC_DIR})

function(download_cusparselt)
  message(
    STATUS
      "Downloading ${CUSPARSELT_URL} to ${CUSPARSELT_DOWNLOAD_DIR}/${CUSPARSELT_CACHE_FILENAME}"
  )
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(
    DOWNLOAD ${CUSPARSELT_URL}
    ${CUSPARSELT_DOWNLOAD_DIR}/${CUSPARSELT_CACHE_FILENAME}
    EXPECTED_MD5 ${CUSPARSELT_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${CUSPARSELT_CACHE_FILENAME} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${CUSPARSELT_CACHE_FILENAME} again"
    )
  endif()
endfunction()

if(EXISTS ${CUSPARSELT_DOWNLOAD_DIR}/${CUSPARSELT_CACHE_FILENAME})
  file(MD5 ${CUSPARSELT_DOWNLOAD_DIR}/${CUSPARSELT_CACHE_FILENAME}
       CUSPARSELT_MD5)
  if(NOT CUSPARSELT_MD5 STREQUAL CUSPARSELT_URL_MD5)
    # clean build file
    file(REMOVE_RECURSE ${CUSPARSELT_PREFIX_DIR})
    file(REMOVE_RECURSE ${CUSPARSELT_INSTALL_DIR})
    download_cusparselt()
  endif()
else()
  download_cusparselt()
endif()

ExternalProject_Add(
  ${CUSPARSELT_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${CUSPARSELT_DOWNLOAD_DIR}/${CUSPARSELT_CACHE_FILENAME}
  PREFIX ${CUSPARSELT_PREFIX_DIR}
  DOWNLOAD_DIR ${CUSPARSELT_DOWNLOAD_DIR}
  DOWNLOAD_NO_PROGRESS 1
  SOURCE_DIR ${CUSPARSELT_INSTALL_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  UPDATE_COMMAND "")

add_library(cusparselt INTERFACE)
add_dependencies(cusparselt ${CUSPARSELT_PROJECT})
set(CUSPARSELT_FOUND ON)
add_definitions(-DPADDLE_WITH_CUSPARSELT)
