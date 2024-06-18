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

set(LAPACK_PREFIX_DIR ${THIRD_PARTY_PATH}/lapack)
set(LAPACK_DOWNLOAD_DIR
    ${PADDLE_SOURCE_DIR}/third_party/lapack/${CMAKE_SYSTEM_NAME})
set(LAPACK_INSTALL_DIR ${THIRD_PARTY_PATH}/install/lapack)
set(LAPACK_LIB_DIR ${LAPACK_INSTALL_DIR}/lib)

# Note(zhouwei): lapack need fortran compiler which many machines don't have, so use precompiled library.
# use lapack tag v3.10.0 on 06/28/2021 https://github.com/Reference-LAPACK/lapack
if(LINUX)
  set(LAPACK_FILE
      "lapack_lnx_v3.10.0.20210628.tar.gz"
      CACHE STRING "" FORCE)
  set(LAPACK_URL
      "https://paddlepaddledeps.bj.bcebos.com/${LAPACK_FILE}"
      CACHE STRING "" FORCE)
  set(LAPACK_URL_MD5 71f8cc8237a8571692f3e07f9a4f25f6)
  set(GNU_RT_LIB_1 "${LAPACK_LIB_DIR}/libquadmath.so.0")
  set(GFORTRAN_LIB "${LAPACK_LIB_DIR}/libgfortran.so.3")
  set(BLAS_LIB "${LAPACK_LIB_DIR}/libblas.so.3")
  set(LAPACK_LIB "${LAPACK_LIB_DIR}/liblapack.so.3")
elseif(WIN32)
  # Refer to [lapack-for-windows] http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke
  set(LAPACK_FILE
      "lapack_win_v3.10.0.20210628.zip"
      CACHE STRING "" FORCE)
  set(LAPACK_URL
      "https://paddlepaddledeps.bj.bcebos.com/${LAPACK_FILE}"
      CACHE STRING "" FORCE)
  set(LAPACK_URL_MD5 590d080392dcd5abbd5dca767a50b63a)
  set(GNU_RT_LIB_1 "${LAPACK_LIB_DIR}/libquadmath-0.dll")
  set(GNU_RT_LIB_2 "${LAPACK_LIB_DIR}/libgcc_s_seh-1.dll")
  set(GFORTRAN_LIB "${LAPACK_LIB_DIR}/libgfortran-3.dll")
  set(BLAS_LIB "${LAPACK_LIB_DIR}/libblas.dll")
  set(LAPACK_LIB "${LAPACK_LIB_DIR}/liblapack.dll")
else() # MacOS
  if(APPLE AND WITH_ARM)
    set(LAPACK_FILE
        "lapack_mac_arm64_v0.3.26.tar.gz"
        CACHE STRING "" FORCE)
    set(LAPACK_URL
        "https://paddlepaddledeps.bj.bcebos.com/${LAPACK_FILE}"
        CACHE STRING "" FORCE)
    set(LAPACK_URL_MD5 3f6412105ae2b7465e5ee90c8673e6d4)
    set(GNU_RT_LIB_1 "${LAPACK_LIB_DIR}/libquadmath.0.dylib")
    set(GNU_RT_LIB_2 "${LAPACK_LIB_DIR}/libgcc_s.1.dylib")
    set(GFORTRAN_LIB "${LAPACK_LIB_DIR}/libgfortran.5.dylib")
    set(BLAS_LIB "${LAPACK_LIB_DIR}/libblas.dylib")
    set(LAPACK_LIB "${LAPACK_LIB_DIR}/liblapack.dylib")
  else()
    set(LAPACK_FILE
        "lapack_mac_v3.10.0.20210628.tar.gz"
        CACHE STRING "" FORCE)
    set(LAPACK_URL
        "https://paddlepaddledeps.bj.bcebos.com/${LAPACK_FILE}"
        CACHE STRING "" FORCE)
    set(LAPACK_URL_MD5 427aecf8dee8523de3566ca8e47944d7)
    set(GNU_RT_LIB_1 "${LAPACK_LIB_DIR}/libquadmath.0.dylib")
    set(GNU_RT_LIB_2 "${LAPACK_LIB_DIR}/libgcc_s.1.dylib")
    set(GFORTRAN_LIB "${LAPACK_LIB_DIR}/libgfortran.5.dylib")
    set(BLAS_LIB "${LAPACK_LIB_DIR}/libblas.3.dylib")
    set(LAPACK_LIB "${LAPACK_LIB_DIR}/liblapack.3.dylib")
  endif()
endif()

function(download_lapack)
  message(
    STATUS "Downloading ${LAPACK_URL} to ${LAPACK_DOWNLOAD_DIR}/${LAPACK_FILE}")
  # NOTE: If the version is updated, consider emptying the folder; maybe add timeout
  file(
    DOWNLOAD ${LAPACK_URL} ${LAPACK_DOWNLOAD_DIR}/${LAPACK_FILE}
    EXPECTED_MD5 ${LAPACK_URL_MD5}
    STATUS ERR)
  if(ERR EQUAL 0)
    message(STATUS "Download ${LAPACK_FILE} success")
  else()
    message(
      FATAL_ERROR
        "Download failed, error: ${ERR}\n You can try downloading ${LAPACK_FILE} again"
    )
  endif()
endfunction()

# Download and check lapack.
if(EXISTS ${LAPACK_DOWNLOAD_DIR}/${LAPACK_FILE})
  file(MD5 ${LAPACK_DOWNLOAD_DIR}/${LAPACK_FILE} LAPACK_MD5)
  if(NOT LAPACK_MD5 STREQUAL LAPACK_URL_MD5)
    # clean build file
    file(REMOVE_RECURSE ${LAPACK_PREFIX_DIR})
    file(REMOVE_RECURSE ${LAPACK_INSTALL_DIR})
    download_lapack()
  endif()
else()
  download_lapack()
endif()

ExternalProject_Add(
  extern_lapack
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${LAPACK_DOWNLOAD_DIR}/${LAPACK_FILE}
  URL_MD5 ${LAPACK_URL_MD5}
  DOWNLOAD_DIR ${LAPACK_DOWNLOAD_DIR}
  SOURCE_DIR ${LAPACK_LIB_DIR}
  PREFIX ${LAPACK_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  PATCH_COMMAND ""
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${BLAS_LIB}
  BUILD_BYPRODUCTS ${LAPACK_LIB})
