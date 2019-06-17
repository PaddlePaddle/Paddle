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

if(NOT ARM_TARGET_OS STREQUAL "armlinux")
    return()
endif()

set(ARMLINUX TRUE)
add_definitions(-DLITE_WITH_LINUX)
set(CMAKE_SYSTEM_NAME Linux)

set(ARMLINUX_ARCH_ABI ${ARM_TARGET_ARCH_ABI} CACHE STRING "Choose Android Arch ABI")

set(ARMLINUX_ARCH_ABI_LIST "armv8" "armv7" "armv7hf")
set_property(CACHE ARMLINUX_ARCH_ABI PROPERTY STRINGS ${ARMLINUX_ARCH_ABI_LIST})
if(NOT ARMLINUX_ARCH_ABI IN_LIST ARMLINUX_ARCH_ABI_LIST)
    message(FATAL_ERROR "ARMLINUX_ARCH_ABI(${ARMLINUX_ARCH_ABI}) must be in one of ${ARMLINUX_ARCH_ABI_LIST}")
endif()

if(ARMLINUX_ARCH_ABI STREQUAL "armv8")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")

    set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
    message(STATUS "NEON is enabled on arm64-v8a")
endif()

if(ARMLINUX_ARCH_ABI STREQUAL "armv7" OR ARMLINUX_ARCH_ABI STREQUAL "armv7hf")
    message(FATAL_ERROR "Not supported building arm linux arm-v7 yet")
endif()

# TODO(TJ): make sure v7 works
if(ARMLINUX_ARCH_ABI STREQUAL "armv7")
    set(CMAKE_SYSTEM_PROCESSOR arm)
    set(CMAKE_C_COMPILER "arm-linux-gnueabi-gcc")
    set(CMAKE_CXX_COMPILER "arm-linux-gnueabi-g++")

    set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}")
    message(STATUS "NEON is enabled on arm-v7a with softfp")
endif()

if(ARMLINUX_ARCH_ABI STREQUAL "armv7hf")
    set(CMAKE_SYSTEM_PROCESSOR arm)
    set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
    set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")

    set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )
    message(STATUS "NEON is enabled on arm-v7a with hard float")
endif()
