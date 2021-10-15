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

# TODO(zhhsplendid): Modify the dir after we have CINN download link
set(CINN_LIB_NAME "libcinnapi.so")
set(CINN_LIB_DIR "/CINN/build/dist/cinn/lib")
set(CINN_INCLUDE_DIR "/CINN/build/dist/cinn/include")


find_library(CINN_LIB_LOCATION NAMES ${CINN_LIB_NAME} PATHS ${CINN_LIB_DIR})
add_library(cinnapi SHARED IMPORTED)
set_target_properties(cinnapi PROPERTIES
	IMPORTED_LOCATION ${CINN_LIB_LOCATION}
	INTERFACE_INCLUDE_DIRECTORIES ${CINN_INCLUDE_DIR})

# Add CINN's dependencies
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
# TODO(zhhsplendid): Modify it
set(ABSL_LIB_DIR "/CINN/build/dist/third_party/absl/lib")
set(ABSL_INCLUDE_DIR "/CINN/build/dist/third_party/absl/include")

add_library(absl STATIC IMPORTED GLOBAL)
set_target_properties(absl PROPERTIES IMPORTED_LOCATION ${ABSL_LIB_DIR}/libabsl_base.a)

foreach(lib_name ${ABSL_LIB_NAMES})
    target_link_libraries(absl INTERFACE ${ABSL_LIB_DIR}/libabsl_${lib_name}.a)
endforeach()
include_directories(${ABSL_INCLUDE_DIR})

# TODO(zhhsplendid): Modify it
set(ISL_LIB_DIR "/CINN/build/dist/third_party/isl/lib")
set(ISL_INCLUDE_DIR "/CINN/build/dist/third_party/isl/include")
add_library(isl STATIC IMPORTED GLOBAL)
set_target_properties(isl PROPERTIES IMPORTED_LOCATION ${ISL_LIB_DIR}/libisl.a)
include_directories(${ISL_INCLUDE_DIR})

set(LLVM_LIB_NAMES
  ExecutionEngine
  )
# TODO(zhhsplendid): Modify it
set(LLVM_LIB_DIR "/CINN/build/dist/third_party/llvm/lib")
set(LLVM_INCLUDE_DIR "/CINN/build/dist/third_party/llvm/include")
add_library(llvm STATIC IMPORTED GLOBAL)
set_target_properties(llvm PROPERTIES IMPORTED_LOCATION ${LLVM_LIB_DIR}/libLLVMCore.a)
foreach(lib_name ${LLVM_LIB_NAMES})
    target_link_libraries(llvm INTERFACE ${LLVM_LIB_DIR}/libLLVM${lib_name}.a)
endforeach()
include_directories(${LLVM_INCLUDE_DIR})
