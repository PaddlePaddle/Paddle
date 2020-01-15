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

set(EIGEN_PREFIX_DIR ${THIRD_PARTY_PATH}/eigen3)
set(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3/src/extern_eigen3)
set(EIGEN_REPOSITORY https://github.com/eigenteam/eigen-git-mirror)
set(EIGEN_TAG        917060c364181f33a735dc023818d5a54f60e54c)

# eigen on cuda9.1 missing header of math_funtions.hpp
# https://stackoverflow.com/questions/43113508/math-functions-hpp-not-found-when-using-cuda-with-eigen
if(WITH_AMD_GPU)
    set(EIGEN_REPOSITORY https://github.com/sabreshao/hipeigen.git)
    set(EIGEN_TAG        7cb2b6e5a4b4a1efe658abb215cd866c6fb2275e)
endif()

cache_third_party(extern_eigen3
    REPOSITORY    ${EIGEN_REPOSITORY}
    TAG           ${EIGEN_TAG}
    DIR           EIGEN_SOURCE_DIR)

if(WIN32)
    file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/Half.h native_src)
    file(TO_NATIVE_PATH ${EIGEN_SOURCE_DIR}/Eigen/src/Core/arch/CUDA/Half.h native_dst)
    set(EIGEN_PATCH_COMMAND copy ${native_src} ${native_dst} /Y)
endif()

set(EIGEN_INCLUDE_DIR ${EIGEN_SOURCE_DIR})
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

if(WITH_AMD_GPU)
    ExternalProject_Add(
        extern_eigen3
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        "${EIGEN_DOWNLOAD_CMD}"
        PREFIX          ${EIGEN_PREFIX_DIR}
        SOURCE_DIR      ${EIGEN_SOURCE_DIR}
        UPDATE_COMMAND    ""
        PATCH_COMMAND   ${EIGEN_PATCH_COMMAND}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
else()
    ExternalProject_Add(
        extern_eigen3
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        "${EIGEN_DOWNLOAD_CMD}"
        PREFIX          ${EIGEN_PREFIX_DIR}
        SOURCE_DIR      ${EIGEN_SOURCE_DIR}
        UPDATE_COMMAND    ""
        PATCH_COMMAND   ${EIGEN_PATCH_COMMAND}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
endif()

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/eigen3_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_eigen3 = \"${dummyfile}\";")
    add_library(eigen3 STATIC ${dummyfile})
else()
    add_library(eigen3 INTERFACE)
endif()

add_dependencies(eigen3 extern_eigen3)
