# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT ANDROID)
    return()
endif()

if(NOT DEFINED ANDROID_NDK)
    set(ANDROID_NDK $ENV{NDK_ROOT})
    if(NOT ANDROID_NDK)
        message(FATAL_ERROR "Must set ANDROID_NDK or env NDK_ROOT")
    endif()
endif()

if(NOT DEFINED ANDROID_ARCH_ABI)
    set(ANDROID_ARCH_ABI "arm64-v8a" CACHE STRING "Choose android platform")
endif()

if(NOT DEFINED ANDROID_API_LEVEL)
    set(ANDROID_API_LEVEL "22")
endif()

if(NOT DEFINED ANDROID_STL_TYPE)
    set(ANDROID_STL_TYPE "c++_static" CACHE STRING "stl type")
endif()

set(ANDROID_ARCH_ABI_LIST "arm64-v8a" "armeabi-v7a" "armeabi-v6" "armeabi"
    "mips" "mips64" "x86" "x86_64")
set_property(CACHE ANDROID_ARCH_ABI PROPERTY STRINGS ${ANDROID_ARCH_ABI_LIST})
if (NOT ANDROID_ARCH_ABI IN_LIST ANDROID_ARCH_ABI_LIST)
    message(FATAL_ERROR "ANDROID_ARCH_ABI must be in one of ${ANDROID_ARCH_ABI_LIST}")
endif()

if(ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
    message(STATUS "NEON is enabled on arm-v7a")
endif()

set(ANDROID_STL_TYPE_LITS "gnustl_static" "c++_static")
set_property(CACHE ANDROID_STL_TYPE PROPERTY STRINGS ${ANDROID_STL_TYPE_LITS}) 
if (NOT ANDROID_STL_TYPE IN_LIST ANDROID_STL_TYPE_LITS)
    message(FATAL_ERROR "ANDROID_STL_TYPE must be in one of ${ANDROID_STL_TYPE_LITS}")
endif()

set(ANDROID_PIE TRUE)
