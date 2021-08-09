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

option(LEAVE_SERVER_ALONE "Try to use a separate serve!" OFF)

if(LEAVE_SERVER_ALONE)
    add_definitions(-DLEAVE_SERVER_ALONE)
endif()

add_subdirectory(impl)

list(APPEND HIERARCHICAL_HCCL_DEPS hierarchical_hccl_impl)

cc_library(hierarchical_hccl SRCS hierarchical_hccl.cc DEPS ${HIERARCHICAL_HCCL_DEPS})
