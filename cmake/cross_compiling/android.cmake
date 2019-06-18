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

if(NOT ARM_TARGET_OS STREQUAL "android")
    return()
endif()

set(ANDROID TRUE)
add_definitions(-DLITE_WITH_LINUX)

if(NOT DEFINED ANDROID_NDK)
    set(ANDROID_NDK $ENV{NDK_ROOT})
    if(NOT ANDROID_NDK)
        message(FATAL_ERROR "Must set ANDROID_NDK or env NDK_ROOT")
    endif()
endif()

if(NOT DEFINED ANDROID_API_LEVEL)
    set(ANDROID_API_LEVEL "22")
endif()

if(NOT DEFINED ANDROID_STL_TYPE)
    set(ANDROID_STL_TYPE "c++_static" CACHE STRING "stl type") # can also use shared
endif()

if(ARM_TARGET_ARCH_ABI STREQUAL "armv7hf")
    message(FATAL_ERROR "ANDROID does not support hardfp on v7 use armv7 instead.")
endif()

set(ANDROID_ARCH_ABI ${ARM_TARGET_ARCH_ABI} CACHE STRING "Choose Android Arch ABI")

if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
    set(ANDROID_ARCH_ABI "arm64-v8a")
endif()

if(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
    set(ANDROID_ARCH_ABI "armeabi-v7a")
endif()

if(ANDROID_ARCH_ABI STREQUAL "armeabi-v7a-softfp")
    set(ANDROID_ARCH_ABI "armeabi-v7a")
endif()

set(ANDROID_ARCH_ABI_LIST "arm64-v8a" "armeabi-v7a" "armeabi-v6" "armeabi"
    "mips" "mips64" "x86" "x86_64")
set_property(CACHE ANDROID_ARCH_ABI PROPERTY STRINGS ${ANDROID_ARCH_ABI_LIST})
if(NOT ANDROID_ARCH_ABI IN_LIST ANDROID_ARCH_ABI_LIST)
    message(FATAL_ERROR "ANDROID_ARCH_ABI must be in one of ${ANDROID_ARCH_ABI_LIST}")
endif()

if(ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
    message(STATUS "armeabi-v7a default use softfp")
    set(CMAKE_ANDROID_ARM_NEON ON)
    message(STATUS "NEON is enabled on arm-v7a with softfp")
endif()

set(ANDROID_STL_TYPE_LITS "gnustl_static" "c++_static")
set_property(CACHE ANDROID_STL_TYPE PROPERTY STRINGS ${ANDROID_STL_TYPE_LITS}) 
if (NOT ANDROID_STL_TYPE IN_LIST ANDROID_STL_TYPE_LITS)
    message(FATAL_ERROR "ANDROID_STL_TYPE must be in one of ${ANDROID_STL_TYPE_LITS}")
endif()

if(ARM_TARGET_LANG STREQUAL "gcc")
    # gcc do not need set lang
    set(ARM_TARGET_LANG "")
endif()

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION ${ANDROID_API_LEVEL})
set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ARCH_ABI})
set(CMAKE_ANDROID_NDK ${ANDROID_NDK})
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION ${ARM_TARGET_LANG})
set(CMAKE_ANDROID_STL_TYPE ${ANDROID_STL_TYPE})

if (ARM_TARGET_LANG STREQUAL "clang")
    if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
        set(triple aarch64-v8a-linux-android)
    elseif(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
        set(triple arm-v7a-linux-android)
    else()
        message(FATAL_ERROR "Clang do not support this ${ARM_TARGET_ARCH_ABI}, use armv8 or armv7")
    endif()

    set(CMAKE_C_COMPILER clang)
    set(CMAKE_C_COMPILER_TARGET ${triple})
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_CXX_COMPILER_TARGET ${triple})

    message(STATUS "CMAKE_CXX_COMPILER_TARGET: ${CMAKE_CXX_COMPILER_TARGET}")
endif()
