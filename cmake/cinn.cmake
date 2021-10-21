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
set(CINN_SOURCE_DIR ${THIRD_PARTY_PATH}/CINN)
# TODO(zhhsplendid): Modify git tag after we have release tag
set(CINN_GIT_TAG 3f004bfa3ed273ecf1de8e7b946433038c79b84f)
set(CINN_OPTIONAL_ARGS -DWITH_CUDA=${WITH_GPU} -DWITH_CUDNN=${WITH_GPU} -DPUBLISH_LIBS=ON)
set(CINN_BUILD_COMMAND $(MAKE) cinncore -j && $(MAKE) cinnapi -j)
ExternalProject_Add(
  external_cinn
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY   "${GIT_URL}/PaddlePaddle/CINN.git"
  GIT_TAG          ${CINN_GIT_TAG}
  PREFIX           ${CINN_SOURCE_DIR}
  UPDATE_COMMAND   ""
  BUILD_COMMAND    ${CINN_BUILD_COMMAND}
  INSTALL_COMMAND  ""
  CMAKE_ARGS       ${CINN_OPTIONAL_ARGS})



ExternalProject_Get_property(external_cinn BINARY_DIR)
ExternalProject_Get_property(external_cinn SOURCE_DIR)
set(CINN_BINARY_DIR ${BINARY_DIR})
set(CINN_SOURCE_DIR ${SOURCE_DIR})

message(STATUS "CINN BINARY_DIR: ${CINN_BINARY_DIR}")
message(STATUS "CINN SOURCE_DIR: ${CINN_SOURCE_DIR}")


#########################
# Add CINN's dependencies
#########################

# Add absl
set(ABSL_LIB_NAMES
  hash
  wyhash
  city
  strings
  throw_delegate
  bad_any_cast_impl
  bad_optional_access
  bad_variant_access
  raw_hash_set
  )
set(ABSL_LIB_DIR "${CINN_BINARY_DIR}/dist/third_party/absl/lib")
set(ABSL_INCLUDE_DIR "${CINN_BINARY_DIR}/dist/third_party/absl/include")
add_library(absl STATIC IMPORTED GLOBAL)
set_target_properties(absl PROPERTIES IMPORTED_LOCATION ${ABSL_LIB_DIR}/libabsl_base.a)
foreach(lib_name ${ABSL_LIB_NAMES})
    target_link_libraries(absl INTERFACE ${ABSL_LIB_DIR}/libabsl_${lib_name}.a)
endforeach()
include_directories(${ABSL_INCLUDE_DIR})

# Add isl
set(ISL_LIB_DIR "${CINN_BINARY_DIR}/dist/third_party/isl/lib")
set(ISL_INCLUDE_DIR "${CINN_BINARY_DIR}/dist/third_party/isl/include")
add_library(isl STATIC IMPORTED GLOBAL)
set_target_properties(isl PROPERTIES IMPORTED_LOCATION ${ISL_LIB_DIR}/libisl.a)
include_directories(${ISL_INCLUDE_DIR})

# Add LLVM
set(LLVM_LIB_NAMES
  ExecutionEngine
  )
set(LLVM_LIB_DIR "${CINN_BINARY_DIR}/dist/third_party/llvm/lib")
set(LLVM_INCLUDE_DIR "${CINN_BINARY_DIR}/dist/third_party/llvm/include")
add_library(llvm STATIC IMPORTED GLOBAL)
set_target_properties(llvm PROPERTIES IMPORTED_LOCATION ${LLVM_LIB_DIR}/libLLVMCore.a)
foreach(lib_name ${LLVM_LIB_NAMES})
    target_link_libraries(llvm INTERFACE ${LLVM_LIB_DIR}/libLLVM${lib_name}.a)
endforeach()
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
add_dependencies(cinn external_cinn absl isl llvm glog gflag)

