# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

# This is a toolchain file for cross-compiling for Raspberry Pi.
# The supported variables are listed belows:
#
# RPI_TOOLCHAIN
# RPI_ARM_NEON
#
# Also you can set CMAKE_C/CXX_COMPILER yourself, through cmake arguments.

IF(NOT RPI)
    return()
ENDIF()
 
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)
SET(CMAKE_SYSTEM_PROCESSOR arm)

# check the exist of raspberry pi toolchain
IF(NOT DEFINED RPI_TOOLCHAIN)
    SET(RPI_TOOLCHAIN $ENV{RPI_TOOLCHAIN}
        CACHE PATH "Folder holds the toolchain of Raspberr Pi")
ENDIF()
IF(NOT RPI_TOOLCHAIN)
    MESSAGE(WARNING "It is recommended to set RPI_TOOLCHAIN to use toolchain.\n"
            "To cross-compile for Raspberry Pi, you need to download the tools using:\n"
            " git clone https://github.com/raspberrypi/tools\n")
ENDIF()

IF(NOT DEFINED RPI_ARM_NEON)
    SET(RPI_ARM_NEON ON)
ENDIF()

IF(RPI_TOOLCHAIN)
    SET(RPI_TOOLCHAIN_ROOT ${RPI_TOOLCHAIN})
    IF(RPI_TOOLCHAIN_ROOT MATCHES "gcc-linaro-arm-linux-gnueabihf-raspbian(-x64)?$")
        # gcc-linaro-arm-linux-gnueabihf-raspbian
        # gcc-linaro-arm-linux-gnueabihf-raspbian-x64
        SET(RPI_TOOLCHAIN_NAME arm-linux-gnueabihf)
    ENDIF()
    SET(RPI_TOOLCHAIN_PREFIX "${RPI_TOOLCHAIN_ROOT}/bin/${RPI_TOOLCHAIN_NAME}-")
ENDIF()

# C compiler
IF(NOT CMAKE_C_COMPILER)
    SET(RPI_C_COMPILER "${RPI_TOOLCHAIN_PREFIX}gcc")
ELSE()
    GET_FILENAME_COMPONENT(RPI_C_COMPILER ${CMAKE_C_COMPILER} PROGRAM)
ENDIF()
IF(NOT EXISTS ${RPI_C_COMPILER})
    MESSAGE(FATAL_ERROR "Cannot find C compiler: ${RPI_C_COMPILER}")
ENDIF()

# CXX compiler
IF(NOT CMAKE_CXX_COMPILER)
    SET(RPI_CXX_COMPILER "${RPI_TOOLCHAIN_PREFIX}g++")
ELSE()
    GET_FILENAME_COMPONENT(RPI_CXX_COMPILER ${CMAKE_CXX_COMPILER} PROGRAM)
ENDIF()
IF(NOT EXISTS ${RPI_CXX_COMPILER})
    MESSAGE(FATAL_ERROR "Cannot find CXX compiler: ${RPI_CXX_COMPILER}")
ENDIF()

SET(CMAKE_C_COMPILER ${RPI_C_COMPILER} CACHE PATH "C compiler" FORCE)
SET(CMAKE_CXX_COMPILER ${RPI_CXX_COMPILER} CACHE PATH "CXX compiler" FORCE)

IF(RPI_ARM_NEON)
    SET(RPI_C_FLAGS "${RPI_C_FLAGS} -mfpu=neon")
ENDIF()

SET(CMAKE_C_FLAGS "${RPI_C_FLAGS} ${CMAKE_C_FLAGS}" CACHE STRING "C flags")
SET(CMAKE_CXX_FLAGS "${RPI_C_FLAGS} ${CMAKE_CXX_FLAGS}" CACHE STRING "CXX flags")
