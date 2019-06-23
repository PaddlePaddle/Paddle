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

INCLUDE(ExternalProject)

SET(OPENCL_CLHPP_SRCS_DIR    ${THIRD_PARTY_PATH}/opencl-clhpp)
SET(OPENCL_CLHPP_INSTALL_DIR ${THIRD_PARTY_PATH}/install/opencl-clhpp)
SET(OPENCL_CLHPP_INCLUDE_DIR "${OPENCL_CLHPP_INSTALL_DIR}" CACHE PATH "opencl-clhpp include directory." FORCE)

INCLUDE_DIRECTORIES(${OPENCL_CLHPP_INCLUDE_DIR})

ExternalProject_Add(
  opencl_clhpp
  GIT_REPOSITORY    "https://github.com/KhronosGroup/OpenCL-CLHPP.git"
  GIT_TAG           "v2.0.10"
  PREFIX            "${OPENCL_CLHPP_SRCS_DIR}"
  CMAKE_ARGS        -DBUILD_DOCS=OFF
                    -DBUILD_EXAMPLES=OFF
                    -DBUILD_TESTS=OFF
                    -DCMAKE_INSTALL_PREFIX=${OPENCL_CLHPP_INSTALL_DIR}
  CMAKE_CACHE_ARGS  -DCMAKE_INSTALL_PREFIX:PATH=${OPENCL_CLHPP_INSTALL_DIR}
                    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)

ADD_DEPENDENCIES(opencl_clhpp opencl_headers)
