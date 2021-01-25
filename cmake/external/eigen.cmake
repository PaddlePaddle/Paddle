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

# update eigen to the commit id 4da2c6b1 on 03/19/2020
set(EIGEN_PREFIX_DIR ${THIRD_PARTY_PATH}/eigen3)
set(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3/src/extern_eigen3)
set(EIGEN_REPOSITORY https://gitlab.com/libeigen/eigen.git)
set(EIGEN_TAG        4da2c6b1974827b1999bab652a3d4703e1992d26)

# the recent version of eigen will cause compilation error on windows
if(WIN32)
    set(EIGEN_REPOSITORY ${GIT_URL}/eigenteam/eigen-git-mirror.git)
    set(EIGEN_TAG        917060c364181f33a735dc023818d5a54f60e54c)
endif()

cache_third_party(extern_eigen3
    REPOSITORY    ${EIGEN_REPOSITORY}
    TAG           ${EIGEN_TAG}
    DIR           EIGEN_SOURCE_DIR)

if(WIN32)
    file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/Half.h native_src)
    file(TO_NATIVE_PATH ${EIGEN_SOURCE_DIR}/Eigen/src/Core/arch/CUDA/Half.h native_dst)
    set(EIGEN_PATCH_COMMAND copy ${native_src} ${native_dst} /Y)
elseif(LINUX)
    # For gxx=4.8, __GXX_ABI_VERSION is less than 1004
    # which will cause a compilation error in Geometry_SSE.h:38:
    # "no matching function for call to 'pmul(Eigen::internal::Packet4f&, __m128)"
    # refer to: https://gitlab.com/libeigen/eigen/-/blob/4da2c6b1974827b1999bab652a3d4703e1992d26/Eigen/src/Core/arch/SSE/PacketMath.h#L33-60
    # add -fabi-version=4 could avoid above error, but will cause "double free corruption" when compile with gcc8
    # so use following patch to solve compilation error with different version of gcc.
    file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/Geometry_SSE.h native_src1)
    file(TO_NATIVE_PATH ${EIGEN_SOURCE_DIR}/Eigen/src/Geometry/arch/Geometry_SSE.h native_dst1)
    # The compiler fully support const expressions since c++14,
    # but Eigen use some const expressions such as std::max and std::min, which are not supported in c++11
    # add patch to avoid compilation error in c++11
    file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/MathFunctions.h native_src2)
    file(TO_NATIVE_PATH ${EIGEN_SOURCE_DIR}/Eigen/src/Core/MathFunctions.h native_dst2)
    if(WITH_ROCM)
        # For HIPCC Eigen::internal::device::numeric_limits is not EIGEN_DEVICE_FUNC
        # which will cause compiler error of using __host__ funciont in __host__ __device__
        file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/Meta.h native_src3)
        file(TO_NATIVE_PATH ${EIGEN_SOURCE_DIR}/Eigen/src/Core/util/Meta.h native_dst3)
        # For HIPCC Eigen::internal::scalar_sum_op<bool,bool> is not EIGEN_DEVICE_FUNC
        # which will cause compiler error of using __host__ funciont in __host__ __device__
        file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/eigen/BinaryFunctors.h native_src4)
        file(TO_NATIVE_PATH ${EIGEN_SOURCE_DIR}/Eigen/src/Core/functors/BinaryFunctors.h native_dst4)
        set(EIGEN_PATCH_COMMAND cp ${native_src1} ${native_dst1} && cp ${native_src2} ${native_dst2} && cp ${native_src3} ${native_dst3} && cp ${native_src4} ${native_dst4})
    else()
        set(EIGEN_PATCH_COMMAND cp ${native_src1} ${native_dst1} && cp ${native_src2} ${native_dst2})
    endif()
endif()

set(EIGEN_INCLUDE_DIR ${EIGEN_SOURCE_DIR})
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

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

add_library(eigen3 INTERFACE)

add_dependencies(eigen3 extern_eigen3)

# sw not support thread_local semantic
if(WITH_SW)
  add_definitions(-DEIGEN_AVOID_THREAD_LOCAL)
endif()
