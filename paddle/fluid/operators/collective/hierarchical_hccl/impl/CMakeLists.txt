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

# Skip build rpath
set(CMAKE_SKIP_BUILD_RPATH True)

# Add definitions ENABLE_DVPP_INTERFACE to use dvpp api
add_definitions(-DENABLE_DVPP_INTERFACE)

# Compile options
add_compile_options(-std=c++11 -fPIC -Wno-aligned-new -Wall -Wextra)

add_subdirectory(rendezvous)

list(APPEND HIERARCHICAL_HCCL_IMPL_DEPS z ssl brpc brpc_store protobuf rendezvous glog gflags leveldb pthread dl rt ascendcl ascend_hccl dynamic_loader dynload_warpctc)
if(WITH_HIERARCHICAL_HCCL)
    list(APPEND HIERARCHICAL_HCCL_IMPL_DEPS crypto)
endif()

cc_library(hierarchical_hccl_impl SRCS hccl_backend.cc hierarchical_backend.cc DEPS ${HIERARCHICAL_HCCL_IMPL_DEPS})
