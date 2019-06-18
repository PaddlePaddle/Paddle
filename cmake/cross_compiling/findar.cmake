# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT ARM_TARGET_LANG STREQUAL "clang")
    # only clang need find ar tool
    return()
endif()

if(NOT EXISTS "${CMAKE_CXX_COMPILER}")
    message(ERROR "Can not find CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER}")
endif()

get_filename_component(AR_PATH ${CMAKE_CXX_COMPILER} PATH)

find_file(AR_TOOL NAMES llvm-ar PATHS ${AR_PATH})

if(NOT AR_TOOL)
    message(ERROR "Failed to find AR_TOOL in ${AR_PATH}")
else()
    set(CMAKE_AR ${AR_TOOL})
    message(STATUS "Found CMAKE_AR : " ${CMAKE_AR})
endif()
