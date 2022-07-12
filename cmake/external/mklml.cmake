# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
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
set(MKLML_INSTALL_DIR ${THIRD_PARTY_PATH}/install/mklml)
set(MKLML_INC_DIR ${MKLML_INSTALL_DIR}/include)
set(MKLML_LIB_DIR ${MKLML_INSTALL_DIR}/lib)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKLML_LIB_DIR}")

if(WIN32)
  set(MKLML_VER
      "mklml_win_2019.0.5.20190502"
      CACHE STRING "" FORCE)
  set(MKLML_URL
      "https://paddlepaddledeps.bj.bcebos.com/${MKLML_VER}.zip"
      CACHE STRING "" FORCE)
  set(MKLML_URL_MD5 ff8c5237570f03eea37377ccfc95a08a)
  set(MKLML_LIB ${MKLML_LIB_DIR}/mklml.lib)
  set(MKLML_IOMP_LIB ${MKLML_LIB_DIR}/libiomp5md.lib)
  set(MKLML_SHARED_LIB ${MKLML_LIB_DIR}/mklml.dll)
  set(MKLML_SHARED_IOMP_LIB ${MKLML_LIB_DIR}/libiomp5md.dll)
else()
  #TODO(intel-huying):
  #  Now enable csrmm function in mklml library temporarily,
  #  it will be updated as offical version later.
  set(MKLML_VER
      "csrmm_mklml_lnx_2019.0.5"
      CACHE STRING "" FORCE)
  set(MKLML_URL
      "http://paddlepaddledeps.bj.bcebos.com/${MKLML_VER}.tgz"
      CACHE STRING "" FORCE)
  set(MKLML_URL_MD5 bc6a7faea6a2a9ad31752386f3ae87da)
  set(MKLML_LIB ${MKLML_LIB_DIR}/libmklml_intel.so)
  set(MKLML_IOMP_LIB ${MKLML_LIB_DIR}/libiomp5.so)
  set(MKLML_SHARED_LIB ${MKLML_LIB_DIR}/libmklml_intel.so)
  set(MKLML_SHARED_IOMP_LIB ${MKLML_LIB_DIR}/libiomp5.so)
endif()

set(MKLML_PROJECT "extern_mklml")
message(STATUS "MKLML_VER: ${MKLML_VER}, MKLML_URL: ${MKLML_URL}")
set(MKLML_PREFIX_DIR ${THIRD_PARTY_PATH}/mklml)
set(MKLML_SOURCE_DIR ${THIRD_PARTY_PATH}/mklml/src/extern_mklml)

# Ninja Generator can not establish the correct dependency relationship
# between the imported library with target, the product file
# in the ExternalProject need to be specified manually, please refer to
# https://stackoverflow.com/questions/54866067/cmake-and-ninja-missing-and-no-known-rule-to-make-it
# It is the same to all other ExternalProject.
ExternalProject_Add(
  ${MKLML_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${MKLML_URL}
  URL_MD5 ${MKLML_URL_MD5}
  PREFIX ${MKLML_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${MKLML_SOURCE_DIR}/include
    ${MKLML_INC_DIR} && ${CMAKE_COMMAND} -E copy_directory
    ${MKLML_SOURCE_DIR}/lib ${MKLML_LIB_DIR}
  BUILD_BYPRODUCTS ${MKLML_LIB}
  BUILD_BYPRODUCTS ${MKLML_IOMP_LIB})

include_directories(${MKLML_INC_DIR})

add_library(mklml SHARED IMPORTED GLOBAL)
set_property(TARGET mklml PROPERTY IMPORTED_LOCATION ${MKLML_LIB})
add_dependencies(mklml ${MKLML_PROJECT})
