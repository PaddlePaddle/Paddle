# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()

cmake_minimum_required(VERSION 3.10)

# define check function
function(check_input_var VAR_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs DEFAULT LIST)
  cmake_parse_arguments(check_input_var "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(var_out "")
  if(NOT DEFINED ${VAR_NAME})
    set(var_out ${check_input_var_DEFAULT})
  else()
    set(var_out ${${VAR_NAME}})
  endif()
  
  if(NOT var_out IN_LIST check_input_var_LIST)
    message(FATAL_ERROR "${VAR_NAME}:${var_out} must be in one of ${check_input_var_LIST}")
  endif()
  set(${VAR_NAME} ${var_out} PARENT_SCOPE)
endfunction(check_input_var)

check_input_var(ARM_TARGET_OS DEFAULT "android" LIST "android" "armlinux")
check_input_var(ARM_TARGET_ARCH_ABI DEFAULT "armv8" LIST "armv8" "armv7" "armv7hf" "arm64-v8a" "armeabi-v7a")
check_input_var(ARM_TARGET_LANG DEFAULT "gcc" LIST "gcc" "clang")
check_input_var(ARM_TARGET_LIB_TYPE DEFAULT "static" LIST "static" "shared")
message(STATUS "Lite ARM Compile ${ARM_TARGET_OS} with ${ARM_TARGET_ARCH_ABI} ${ARM_TARGET_LANG}")

include(cross_compiling/host)
include(cross_compiling/armlinux)
include(cross_compiling/android)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default use Release in android" FORCE)
endif()

if(NOT THIRD_PARTY_BUILD_TYPE)
    set(THIRD_PARTY_BUILD_TYPE "MinSizeRel" CACHE STRING "Default use MinSizeRel in android" FORCE)
endif()

