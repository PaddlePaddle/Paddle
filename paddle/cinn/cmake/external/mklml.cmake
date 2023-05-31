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

if(NOT ${WITH_MKL_CBLAS})
  return()
endif()

# IF(APPLE)
#     MESSAGE(WARNING "Mac is not supported with MKLML in Paddle yet. Force WITH_MKLML=OFF.")
#     SET(WITH_MKLML OFF CACHE STRING "Disable MKLML package in MacOS" FORCE)
#     return()
# ENDIF()

include(ExternalProject)
set(MKLML_DST_DIR "mklml")
set(MKLML_INSTALL_ROOT "${THIRD_PARTY_PATH}/install")
set(MKLML_INSTALL_DIR ${MKLML_INSTALL_ROOT}/${MKLML_DST_DIR})
set(MKLML_ROOT ${MKLML_INSTALL_DIR})
set(MKLML_INC_DIR ${MKLML_ROOT}/include)
set(MKLML_LIB_DIR ${MKLML_ROOT}/lib)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKLML_ROOT}/lib")

set(TIME_VERSION "2019.0.1.20181227")
if(WIN32)
  set(MKLML_VER
      "mklml_win_${TIME_VERSION}"
      CACHE STRING "" FORCE)
  set(MKLML_URL
      "http://paddlepaddledeps.bj.bcebos.com/${MKLML_VER}.zip"
      CACHE STRING "" FORCE)
  set(MKLML_LIB ${MKLML_LIB_DIR}/mklml.lib)
  set(MKLML_IOMP_LIB ${MKLML_LIB_DIR}/libiomp5md.lib)
  set(MKLML_SHARED_LIB ${MKLML_LIB_DIR}/mklml.dll)
  set(MKLML_SHARED_IOMP_LIB ${MKLML_LIB_DIR}/libiomp5md.dll)
else()
  #TODO(intel-huying):
  #  Now enable Erf function in mklml library temporarily, it will be updated as offical version later.
  set(MKLML_VER
      "Glibc225_vsErf_mklml_lnx_${TIME_VERSION}"
      CACHE STRING "" FORCE)
  set(MKLML_URL
      "http://paddlepaddledeps.bj.bcebos.com/${MKLML_VER}.tgz"
      CACHE STRING "" FORCE)
  set(MKLML_LIB ${MKLML_LIB_DIR}/libmklml_intel.so)
  set(MKLML_IOMP_LIB ${MKLML_LIB_DIR}/libiomp5.so)
  set(MKLML_SHARED_LIB ${MKLML_LIB_DIR}/libmklml_intel.so)
  set(MKLML_SHARED_IOMP_LIB ${MKLML_LIB_DIR}/libiomp5.so)
endif()

set(MKLML_PROJECT "extern_mklml")
message(STATUS "MKLML_VER: ${MKLML_VER}, MKLML_URL: ${MKLML_URL}")
set(MKLML_SOURCE_DIR "${THIRD_PARTY_PATH}/mklml")
set(MKLML_DOWNLOAD_DIR "${MKLML_SOURCE_DIR}/src/${MKLML_PROJECT}")

ExternalProject_Add(
  ${MKLML_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${MKLML_SOURCE_DIR}
  URL ${MKLML_URL}
  DOWNLOAD_DIR ${MKLML_DOWNLOAD_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${MKLML_DOWNLOAD_DIR}/include
    ${MKLML_INC_DIR} && ${CMAKE_COMMAND} -E copy_directory
    ${MKLML_DOWNLOAD_DIR}/lib ${MKLML_LIB_DIR}
  BUILD_BYPRODUCTS ${MKLML_SHARED_LIB}
  BUILD_BYPRODUCTS ${MKLML_SHARED_IOMP_LIB})

include_directories(${MKLML_INC_DIR})

set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/mklml_dummy.c)

if(NOT EXISTS ${dummyfile})
  file(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")
endif()
add_library(mklml STATIC ${dummyfile})
add_definitions(-DCINN_WITH_MKL_CBLAS)

target_link_libraries(mklml ${MKLML_LIB} ${MKLML_IOMP_LIB})
add_dependencies(mklml ${MKLML_PROJECT})
