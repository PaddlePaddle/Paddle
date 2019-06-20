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

SET(OPENCL_HEADERS_SRCS_DIR    ${THIRD_PARTY_PATH}/opencl-headers)
SET(OPENCL_HEADERS_INCLUDE_DIR "${OPENCL_HEADERS_SRCS_DIR}/src/opencl_headers" CACHE PATH "opencl-headers include directory." FORCE)

INCLUDE_DIRECTORIES(${OPENCL_HEADERS_INCLUDE_DIR})

ExternalProject_Add(
  opencl_headers
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY    "https://github.com/KhronosGroup/OpenCL-Headers.git"
  GIT_TAG           "c5a4bbeabb10d8ed3d1c651b93aa31737bc473dd"
  PREFIX            ${OPENCL_HEADERS_SRCS_DIR}
  DOWNLOAD_NAME     "OpenCL-Headers"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
