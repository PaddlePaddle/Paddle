# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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

# This is a toolchain file for cross-compiling for Android, and the
# configuration refers to the open-source resposity:
#     https://github.com/taka-no-me/android-cmake
# Most of the variables are compatible with that used in
#     https://developer.android.com/ndk/guides/cmake.html
# The supported variables are listed belows:
# 
# ANDROID_STANDALONE_TOOLCHAIN
# ANDROID_ABI
# ANDROID_NATIVE_API_LEVEL
# ANDROID_ARM_MODE
# ANDROID_ARM_NEON
#
# For CMake >= 3.7.0, all the settings will be delivered to CMake system
# variables to let CMake do the cross-compiling configurations itself.
# More detail of cross-compiling settings
#     https://cmake.org/cmake/help/v3.7/manual/cmake-toolchains.7.html

IF(NOT ANDROID)
    return()
ENDIF()

# check the exist of android standalone toolchain
IF(NOT DEFINED ANDROID_STANDALONE_TOOLCHAIN)
    SET(ANDROID_STANDALONE_TOOLCHAIN $ENV{ANDROID_STANDALONE_TOOLCHAIN}
        CACHE PATH "Folder holds the standalone toolchain of Android NDK")
ENDIF()
IF(NOT ANDROID_STANDALONE_TOOLCHAIN)
    MESSAGE(WARNING "It is recommended to set ANDROID_STANDALONE_TOOLCHAIN to "
            "use a standalone toolchain.\n"
            "To cross-compile for Android, you need to:\n"
            "1. Download an Android NDK from"
            " https://developer.android.com/ndk/downloads/index.html\n"
            "2. Setup a standalone toolchain"
            "https://developer.android.google.cn/ndk/guides/standalone_toolchain.html?hl=zh-cn\n")
ENDIF()

IF(NOT DEFINED CMAKE_SYSTEM_VERSION AND ANDROID_NATIVE_API_LEVEL)
    IF(ANDROID_NATIVE_API_LEVEL MATCHES "^android-[0-9]+$")
        STRING(REPLACE "android-" "" CMAKE_SYSTEM_VERSION "${CMAKE_MATCH_0}")
    ELSEIF(ANDROID_NATIVE_API_LEVEL MATCHES "^[0-9]+$")
        SET(CMAKE_SYSTEM_VERSION ${ANDROID_NATIVE_API_LEVEL})
    ENDIF()
ENDIF()

IF(NOT DEFINED ANDROID_ABI)
    SET(ANDROID_ABI "armeabi-v7a")
ENDIF()

IF(NOT DEFINED ANDROID_ARM_MODE)
    SET(ANDROID_ARM_MODE ON)
ENDIF()
IF(ANDROID_ARM_MODE)
    SET(ANDROID_ARM_MODE_NAME "arm")
ELSE(ANDROID_ARM_MODE)
    SET(ANDROID_ARM_MODE_NAME "thumb")
ENDIF(ANDROID_ARM_MODE)

IF(NOT DEFINED ANDROID_ARM_NEON)
    SET(ANDROID_ARM_NEON ON)
ENDIF()

IF("${CMAKE_VERSION}" VERSION_LESS "3.7.0")
    IF("${CMAKE_VERSION}" VERSION_LESS "3.1.0")
        SET(CMAKE_SYSTEM_NAME "Linux")
    ENDIF()
    MESSAGE(WARNING "It is recommended to use CMake >= 3.7.0 (current version: "
            "${CMAKE_VERSION}), when cross-compiling for Android.")

    IF(ANDROID_STANDALONE_TOOLCHAIN)
        SET(CMAKE_SYSROOT "${ANDROID_STANDALONE_TOOLCHAIN}/sysroot")

        IF(NOT CMAKE_SYSTEM_VERSION)
            SET(ANDROID_STANDALONE_TOOLCHAIN_API "")
            SET(ANDROID_API_LEVEL_H_REGEX "^[\t ]*#[\t ]*define[\t ]+__ANDROID_API__[\t ]+([0-9]+)")
            FILE(STRINGS "${ANDROID_STANDALONE_TOOLCHAIN}/sysroot/usr/include/android/api-level.h"
                ANDROID_API_LEVEL_H_CONTENT REGEX "${ANDROID_API_LEVEL_H_REGEX}")
            IF(ANDROID_API_LEVEL_H_CONTENT MATCHES "${ANDROID_API_LEVEL_H_REGEX}")
                SET(ANDROID_STANDALONE_TOOLCHAIN_API "${CMAKE_MATCH_1}")
            ENDIF()
            SET(CMAKE_SYSTEM_VERSION ${ANDROID_STANDALONE_TOOLCHAIN_API})
        ENDIF()

        # Toolchain
        SET(ANDROID_TOOLCHAIN "gcc")
        SET(ANDROID_TOOLCHAIN_ROOT ${ANDROID_STANDALONE_TOOLCHAIN})
        IF(ANDROID_ABI MATCHES "^armeabi(-v7a)?$")
            SET(ANDROID_TOOLCHAIN_NAME arm-linux-androideabi)
            IF(ANDROID_ABI STREQUAL "armeabi")
                SET(CMAKE_SYSTEM_PROCESSOR armv5te)
            ELSEIF(ANDROID_ABI STREQUAL "armeabi-v7a")
                SET(CMAKE_SYSTEM_PROCESSOR armv7-a)
            ENDIF()
        ENDIF()
        IF(ANDROID_ABI STREQUAL "arm64-v8a")
            SET(ANDROID_TOOLCHAIN_NAME aarch64-linux-android)
        ENDIF()
        SET(ANDROID_TOOLCHAIN_PREFIX "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_NAME}-")
    ENDIF()

    # C compiler
    IF(NOT CMAKE_C_COMPILER)
        SET(ANDROID_C_COMPILER "${ANDROID_TOOLCHAIN_PREFIX}gcc")
    ELSE()
        GET_FILENAME_COMPONENT(ANDROID_C_COMPILER ${CMAKE_C_COMPILER} PROGRAM)
    ENDIF()
    IF(NOT EXISTS ${ANDROID_C_COMPILER})
        MESSAGE(FATAL_ERROR "Cannot find C compiler: ${ANDROID_C_COMPILER}")
    ENDIF()

    # CXX compiler
    IF(NOT CMAKE_CXX_COMPILER)
        SET(ANDROID_CXX_COMPILER "${ANDROID_TOOLCHAIN_PREFIX}g++")
    ELSE()
        GET_FILENAME_COMPONENT(ANDROID_CXX_COMPILER ${CMAKE_CXX_COMPILER} PROGRAM)
    ENDIF()
    IF(NOT EXISTS ${ANDROID_CXX_COMPILER})
        MESSAGE(FATAL_ERROR "Cannot find CXX compiler: ${ANDROID_CXX_COMPILER}")
    ENDIF()

    SET(CMAKE_C_COMPILER ${ANDROID_C_COMPILER} CACHE PATH "C compiler" FORCE)
    SET(CMAKE_CXX_COMPILER ${ANDROID_CXX_COMPILER} CACHE PATH "CXX compiler" FORCE)

    # Toolchain and ABI specific flags.
    SET(ANDROID_COMPILER_FLAGS "-ffunction-sections -fdata-sections -finline-limit=64")
    SET(ANDROID_LINKER_FLAGS "-Wl,--gc-sections")

    IF(ANDROID_ABI STREQUAL "armeabi")
        LIST(APPEND ANDROID_COMPILER_FLAGS
             -march=armv5te
             -mtune=xscale
             -msoft-float)
    ENDIF()
    IF(ANDROID_ABI STREQUAL "armeabi-v7a")
        LIST(APPEND ANDROID_COMPILER_FLAGS
             -march=armv7-a
             -mfloat-abi=softfp)
        IF(ANDROID_ARM_NEON)
            LIST(APPEND ANDROID_COMPILER_FLAGS -mfpu=neon)
        ELSE()
            LIST(APPEND ANDROID_COMPILER_FLAGS -mfpu=vfpv3-d16)
        ENDIF()
        LIST(APPEND ANDROID_LINKER_FLAGS -Wl,--fix-cortex-a8)
    ENDIF()

    IF(ANDROID_ABI MATCHES "^armeabi(-v7a)?$")
        IF(ANDROID_ARM_MODE)
            LIST(APPEND ANDROID_COMPILER_FLAGS -marm)
        ELSE()
            LIST(APPEND ANDROID_COMPILER_FLAGS -mthumb)
        ENDIF()
    ENDIF()

    IF(ANDROID_ABI STREQUAL "arm64-v8a")
      LIST(APPEND ANDROID_COMPILER_FLAGS -march=armv8-a)
    ENDIF()

    STRING(REPLACE ";" " " ANDROID_COMPILER_FLAGS "${ANDROID_COMPILER_FLAGS}")
    STRING(REPLACE ";" " " ANDROID_LINKER_FLAGS "${ANDROID_LINKER_FLAGS}")

    SET(CMAKE_C_FLAGS "${ANDROID_COMPILER_FLAGS} ${CMAKE_C_FLAGS}"
        CACHE STRING "C flags")
    SET(CMAKE_CXX_FLAGS "${ANDROID_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}"
        CACHE STRING "CXX flags")
    SET(CMAKE_SHARED_LINKER_FLAGS "${ANDROID_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}"
        CACHE STRING "shared linker flags")

    SET(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
    SET(CMAKE_EXE_LINKER_FLAGS "-pie -fPIE ${ANDROID_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}"
        CACHE STRING "executable linker flags")

    MESSAGE(STATUS "Android: Targeting API '${CMAKE_SYSTEM_VERSION}' "
            "with architecture '${ANDROID_ARM_MODE_NAME}', "
            "ABI '${ANDROID_ABI}', and processor '${CMAKE_SYSTEM_PROCESSOR}'")
    MESSAGE(STATUS "System CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS})
    MESSAGE(STATUS "System CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
ELSE()
    IF(ANDROID_STANDALONE_TOOLCHAIN)
        SET(CMAKE_ANDROID_STANDALONE_TOOLCHAIN ${ANDROID_STANDALONE_TOOLCHAIN})
    ENDIF()
    SET(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
    SET(CMAKE_ANDROID_ARM_MODE ${ANDROID_ARM_MODE})
    SET(CMAKE_ANDROID_ARM_NEON ${ANDROID_ARM_NEON})
ENDIF()
