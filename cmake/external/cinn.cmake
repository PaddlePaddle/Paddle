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

if (NOT WITH_CINN)
  return()
endif()

# TODO(zhhsplendid): CINN has lots of warnings during early development.
# They will be treated as errors under paddle. We set no-error now and we will
# clean the code in the future.
add_definitions(-w)

######################################
# Build CINN from Git External Project
######################################
include(ExternalProject)
set(CINN_PREFIX_DIR ${THIRD_PARTY_PATH}/CINN)
set(CINN_GIT_TAG release/v0.2)
set(CINN_OPTIONAL_ARGS -DPY_VERSION=${PY_VERSION}
                       -DWITH_CUDA=${WITH_GPU}
                       -DWITH_CUDNN=${WITH_GPU}
                       -DWITH_MKL_CBLAS=${WITH_MKL}
                       -DWITH_MKLDNN=${WITH_MKL}
                       -DPUBLISH_LIBS=ON
                       -DWITH_TESTING=ON
)
set(CINN_BUILD_COMMAND $(MAKE) cinnapi -j)
ExternalProject_Add(
  external_cinn
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY   "${GIT_URL}/PaddlePaddle/CINN.git"
  GIT_TAG          ${CINN_GIT_TAG}
  PREFIX           ${CINN_PREFIX_DIR}
  BUILD_COMMAND    ${CINN_BUILD_COMMAND}
  INSTALL_COMMAND  ""
  CMAKE_ARGS       ${CINN_OPTIONAL_ARGS})



ExternalProject_Get_property(external_cinn BINARY_DIR)
ExternalProject_Get_property(external_cinn SOURCE_DIR)
set(CINN_BINARY_DIR ${BINARY_DIR})
set(CINN_SOURCE_DIR ${SOURCE_DIR})

message(STATUS "CINN BINARY_DIR: ${CINN_BINARY_DIR}")
message(STATUS "CINN SOURCE_DIR: ${CINN_SOURCE_DIR}")


######################################
# Add CINN's dependencies header files
######################################

# Add absl
set(ABSL_INCLUDE_DIR "${CINN_BINARY_DIR}/dist/third_party/absl/include")
include_directories(${ABSL_INCLUDE_DIR})

# Add isl
set(ISL_INCLUDE_DIR "${CINN_BINARY_DIR}/dist/third_party/isl/include")
include_directories(${ISL_INCLUDE_DIR})

# Add LLVM
set(LLVM_INCLUDE_DIR "${CINN_BINARY_DIR}/dist/third_party/llvm/include")
include_directories(${LLVM_INCLUDE_DIR})

######################################################
# Put external_cinn and dependencies together as a lib
######################################################

set(CINN_LIB_NAME "libcinnapi.so")
set(CINN_LIB_LOCATION "${CINN_BINARY_DIR}/dist/cinn/lib")
set(CINN_INCLUDE_DIR "${CINN_BINARY_DIR}/dist/cinn/include")

add_library(cinn SHARED IMPORTED GLOBAL)
set_target_properties(cinn PROPERTIES IMPORTED_LOCATION "${CINN_LIB_LOCATION}/${CINN_LIB_NAME}")
include_directories(${CINN_INCLUDE_DIR})
add_dependencies(cinn external_cinn)
