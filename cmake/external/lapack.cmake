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

INCLUDE (ExternalProject)

SET(LAPACK_PREFIX_DIR       ${THIRD_PARTY_PATH}/lapack)
SET(LAPACK_SOURCE_DIR       ${THIRD_PARTY_PATH}/lapack/src/extern_lapack)
SET(LAPACK_INSTALL_DIR      ${THIRD_PARTY_PATH}/install/lapack)
SET(LAPACK_LIB_DIR          ${LAPACK_INSTALL_DIR}/lib)

# Note(zhouwei): lapack need fortan compiler which many machines don't have, so use precompiled library.
# use lapack tag v3.10.0 on 06/28/2021 https://github.com/Reference-LAPACK/lapack
if(LINUX)
    SET(LAPACK_VER  "lapack_lnx_v3.10.0.20210628" CACHE STRING "" FORCE)
    SET(LAPACK_URL  "https://paddlepaddledeps.bj.bcebos.com/${LAPACK_VER}.tar.gz" CACHE STRING "" FORCE)
    SET(LAPACK_URL_MD5          71f8cc8237a8571692f3e07f9a4f25f6)
    SET(GNU_RT_LIB_1            "${LAPACK_LIB_DIR}/libquadmath.so.0")
    SET(GFORTRAN_LIB            "${LAPACK_LIB_DIR}/libgfortran.so.3")
    SET(BLAS_LIB                "${LAPACK_LIB_DIR}/libblas.so.3")
    SET(LAPACK_LIB              "${LAPACK_LIB_DIR}/liblapack.so.3")
elseif(WIN32)
    # Refer to [lapack-for-windows] http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke
    SET(LAPACK_VER  "lapack_win_v3.10.0.20210628" CACHE STRING "" FORCE)
    SET(LAPACK_URL  "https://paddlepaddledeps.bj.bcebos.com/${LAPACK_VER}.zip" CACHE STRING "" FORCE)
    SET(LAPACK_URL_MD5          590d080392dcd5abbd5dca767a50b63a)
    SET(GNU_RT_LIB_1            "${LAPACK_LIB_DIR}/libquadmath-0.dll")
    SET(GNU_RT_LIB_2            "${LAPACK_LIB_DIR}/libgcc_s_seh-1.dll")
    SET(GFORTRAN_LIB            "${LAPACK_LIB_DIR}/libgfortran-3.dll")
    SET(BLAS_LIB                "${LAPACK_LIB_DIR}/libblas.dll")
    SET(LAPACK_LIB              "${LAPACK_LIB_DIR}/liblapack.dll")
else()
    SET(LAPACK_VER  "lapack_mac_v3.10.0.20210628" CACHE STRING "" FORCE)
    SET(LAPACK_URL  "https://paddlepaddledeps.bj.bcebos.com/${LAPACK_VER}.tar.gz" CACHE STRING "" FORCE)
    SET(LAPACK_URL_MD5          427aecf8dee8523de3566ca8e47944d7)
    SET(GNU_RT_LIB_1            "${LAPACK_LIB_DIR}/libquadmath.0.dylib")
    SET(GNU_RT_LIB_2            "${LAPACK_LIB_DIR}/libgcc_s.1.dylib")
    SET(GFORTRAN_LIB            "${LAPACK_LIB_DIR}/libgfortran.5.dylib")
    SET(BLAS_LIB                "${LAPACK_LIB_DIR}/libblas.3.dylib")
    SET(LAPACK_LIB              "${LAPACK_LIB_DIR}/liblapack.3.dylib")
endif()

ExternalProject_Add(
    extern_lapack
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL                     ${LAPACK_URL}
    URL_MD5                 ${LAPACK_URL_MD5}
    PREFIX                  ${LAPACK_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS    1
    PATCH_COMMAND           ""
    UPDATE_COMMAND          ""
    CONFIGURE_COMMAND       ""
    BUILD_COMMAND           ""
    INSTALL_COMMAND         ${CMAKE_COMMAND} -E copy_directory ${LAPACK_SOURCE_DIR} ${LAPACK_LIB_DIR}
    BUILD_BYPRODUCTS        ${BLAS_LIB}
    BUILD_BYPRODUCTS        ${LAPACK_LIB}
)
