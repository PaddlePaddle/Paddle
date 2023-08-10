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

set(BOOST_PROJECT "extern_boost")
# To release PaddlePaddle as a pip package, we have to follow the
# manylinux1 standard, which features as old Linux kernels and
# compilers as possible and recommends CentOS 5. Indeed, the earliest
# CentOS version that works with NVIDIA CUDA is CentOS 6.  And a new
# version of boost, say, 1.66.0, doesn't build on CentOS 6.  We
# checked that the devtools package of CentOS 6 installs boost 1.41.0.
# So we use 1.41.0 here.
set(BOOST_VER "1.41.0")
set(BOOST_FILE
    "boost_1_41_0.tar.gz"
    CACHE STRING "" FORCE)
set(BOOST_DOWNLOAD_URL "http://paddlepaddledeps.bj.bcebos.com/${BOOST_FILE}")

set(BOOST_URL_MD5 f891e8c2c9424f0565f0129ad9ab4aff)
set(BOOST_DOWNLOAD_DIR ${PADDLE_SOURCE_DIR}/third_party/boost)
set(BOOST_PREFIX_DIR ${THIRD_PARTY_PATH}/boost)
set(BOOST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/boost)
message(STATUS "BOOST_TAR: ${BOOST_TAR}, BOOST_URL: ${BOOST_URL}")

set(BOOST_INCLUDE_DIR
    "${BOOST_DOWNLOAD_DIR}"
    CACHE PATH "boost include directory." FORCE)
set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)
include_directories(${BOOST_INCLUDE_DIR})

function(download_boost)
  message(
    STATUS
      "Downloading ${BOOST_DOWNLOAD_URL} to ${BOOST_DOWNLOAD_DIR}/${BOOST_FILE}"
  )
  file(
    DOWNLOAD ${BOOST_DOWNLOAD_URL} ${BOOST_DOWNLOAD_DIR}/${BOOST_FILE}
    EXPECTED_MD5 ${BOOST_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${BOOST_FILE} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${BOOST_FILE} again"
    )
  endif()
endfunction()

# Download and check boost.
if(EXISTS ${BOOST_DOWNLOAD_DIR}/${BOOST_FILE})
  file(MD5 ${BOOST_DOWNLOAD_DIR}/${BOOST_FILE} BOOST_MD5)
  if(NOT BOOST_MD5 STREQUAL BOOST_URL_MD5)
    # clean build file
    file(REMOVE_RECURSE ${BOOST_PREFIX_DIR})
    file(REMOVE_RECURSE ${BOOST_INSTALL_DIR})
    download_boost()
  endif()
else()
  download_boost()
endif()

ExternalProject_Add(
  ${BOOST_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${BOOST_DOWNLOAD_DIR}/${BOOST_FILE}
  URL_MD5 ${BOOST_URL_MD5}
  DOWNLOAD_DIR ${BOOST_DOWNLOAD_DIR}
  PREFIX ${BOOST_PREFIX_DIR}
  SOURCE_DIR ${BOOST_INSTALL_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  UPDATE_COMMAND "")

if(${CMAKE_VERSION} VERSION_LESS "3.3.0" OR NOT WIN32)
  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/boost_dummy.c)
  file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
  add_library(boost STATIC ${dummyfile})
else()
  add_library(boost INTERFACE)
endif()

add_dependencies(boost ${BOOST_PROJECT})
set(Boost_INCLUDE_DIR ${BOOST_INCLUDE_DIR})
