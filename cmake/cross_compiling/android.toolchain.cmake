# Copyright (C) 2016 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configurable variables.
# Modeled after the ndk-build system.
# For any variables defined in:
#         https://developer.android.com/ndk/guides/android_mk.html
#         https://developer.android.com/ndk/guides/application_mk.html
# if it makes sense for CMake, then replace LOCAL, APP, or NDK with ANDROID, and
# we have that variable below.
# The exception is ANDROID_TOOLCHAIN vs NDK_TOOLCHAIN_VERSION.
# Since we only have one version of each gcc and clang, specifying a version
# doesn't make much sense.
#
# ANDROID_TOOLCHAIN
# ANDROID_ABI
# ANDROID_PLATFORM
# ANDROID_STL
# ANDROID_PIE
# ANDROID_CPP_FEATURES
# ANDROID_ALLOW_UNDEFINED_SYMBOLS
# ANDROID_ARM_MODE
# ANDROID_ARM_NEON
# ANDROID_DISABLE_NO_EXECUTE
# ANDROID_DISABLE_RELRO
# ANDROID_DISABLE_FORMAT_STRING_CHECKS
# ANDROID_CCACHE

# cmake_minimum_required(VERSION 3.6.0)

# Inhibit all of CMake's own NDK handling code.
set(CMAKE_SYSTEM_VERSION 1)

# CMake invokes the toolchain file twice during the first build, but only once
# during subsequent rebuilds. This was causing the various flags to be added
# twice on the first build, and on a rebuild ninja would see only one set of the
# flags and rebuild the world.
# https://github.com/android-ndk/ndk/issues/323
if(ANDROID_NDK_TOOLCHAIN_INCLUDED)
  return()
endif(ANDROID_NDK_TOOLCHAIN_INCLUDED)
set(ANDROID_NDK_TOOLCHAIN_INCLUDED true)

# Android NDK
if(NOT ANDROID_NDK)
  get_filename_component(ANDROID_NDK "$ENV{NDK_ROOT}" ABSOLUTE)
else()
  # Allow the user to specify their own NDK path, but emit a warning. This is an
  # uncommon use case, but helpful if users want to use a bleeding edge
  # toolchain file with a stable NDK.
  # https://github.com/android-ndk/ndk/issues/473
  message(WARNING "Using custom NDK path (ANDROID_NDK is set): ${ANDROID_NDK}")
endif()
file(TO_CMAKE_PATH "${ANDROID_NDK}" ANDROID_NDK)

# Android NDK revision
message("${ANDROID_NDK}")

file(READ "${ANDROID_NDK}/source.properties" ANDROID_NDK_SOURCE_PROPERTIES)
set(ANDROID_NDK_SOURCE_PROPERTIES_REGEX
  "^Pkg\\.Desc = Android NDK\nPkg\\.Revision = ([0-9]+)\\.")
if(NOT ANDROID_NDK_SOURCE_PROPERTIES MATCHES "${ANDROID_NDK_SOURCE_PROPERTIES_REGEX}")
  message(SEND_ERROR "Failed to parse Android NDK revision: ${ANDROID_NDK}/source.properties.\n${ANDROID_NDK_SOURCE_PROPERTIES}")
endif()
string(REGEX REPLACE "${ANDROID_NDK_SOURCE_PROPERTIES_REGEX}" "\\1"
  ANDROID_NDK_REVISION "${ANDROID_NDK_SOURCE_PROPERTIES}")

# Touch toolchain variable to suppress "unused variable" warning.
# This happens if CMake is invoked with the same command line the second time.
if(CMAKE_TOOLCHAIN_FILE)
endif()

# Compatibility for configurable variables.
# Compatible with configurable variables from the other toolchain file:
#         https://github.com/taka-no-me/android-cmake
# TODO: We should consider dropping compatibility to simplify things once most
# of our users have migrated to our standard set of configurable variables.
if(ANDROID_TOOLCHAIN_NAME AND NOT ANDROID_TOOLCHAIN)
  if(ANDROID_TOOLCHAIN_NAME MATCHES "-clang([0-9].[0-9])?$")
    set(ANDROID_TOOLCHAIN clang)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "-[0-9].[0-9]$")
    set(ANDROID_TOOLCHAIN gcc)
  endif()
endif()
if(ANDROID_ABI STREQUAL "armeabi-v7a with NEON")
  set(ANDROID_ABI armeabi-v7a)
  set(ANDROID_ARM_NEON TRUE)
elseif(ANDROID_TOOLCHAIN_NAME AND NOT ANDROID_ABI)
  if(ANDROID_TOOLCHAIN_NAME MATCHES "^arm-linux-androideabi-")
    set(ANDROID_ABI armeabi-v7a)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^aarch64-linux-android-")
    set(ANDROID_ABI arm64-v8a)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^x86-")
    set(ANDROID_ABI x86)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^x86_64-")
    set(ANDROID_ABI x86_64)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^mipsel-linux-android-")
    set(ANDROID_ABI mips)
  elseif(ANDROID_TOOLCHAIN_NAME MATCHES "^mips64el-linux-android-")
    set(ANDROID_ABI mips64)
  endif()
endif()
if(ANDROID_NATIVE_API_LEVEL AND NOT ANDROID_PLATFORM)
  if(ANDROID_NATIVE_API_LEVEL MATCHES "^android-[0-9]+$")
    set(ANDROID_PLATFORM ${ANDROID_NATIVE_API_LEVEL})
  elseif(ANDROID_NATIVE_API_LEVEL MATCHES "^[0-9]+$")
    set(ANDROID_PLATFORM android-${ANDROID_NATIVE_API_LEVEL})
  endif()
endif()
if(DEFINED ANDROID_APP_PIE AND NOT DEFINED ANDROID_PIE)
  set(ANDROID_PIE "${ANDROID_APP_PIE}")
endif()
if(ANDROID_STL_FORCE_FEATURES AND NOT DEFINED ANDROID_CPP_FEATURES)
  set(ANDROID_CPP_FEATURES "rtti exceptions")
endif()
if(DEFINED ANDROID_NO_UNDEFINED AND NOT DEFINED ANDROID_ALLOW_UNDEFINED_SYMBOLS)
  if(ANDROID_NO_UNDEFINED)
    set(ANDROID_ALLOW_UNDEFINED_SYMBOLS FALSE)
  else()
    set(ANDROID_ALLOW_UNDEFINED_SYMBOLS TRUE)
  endif()
endif()
if(DEFINED ANDROID_SO_UNDEFINED AND NOT DEFINED ANDROID_ALLOW_UNDEFINED_SYMBOLS)
  set(ANDROID_ALLOW_UNDEFINED_SYMBOLS "${ANDROID_SO_UNDEFINED}")
endif()
if(DEFINED ANDROID_FORCE_ARM_BUILD AND NOT ANDROID_ARM_MODE)
  if(ANDROID_FORCE_ARM_BUILD)
    set(ANDROID_ARM_MODE arm)
  else()
    set(ANDROID_ARM_MODE thumb)
  endif()
endif()
if(DEFINED ANDROID_NOEXECSTACK AND NOT DEFINED ANDROID_DISABLE_NO_EXECUTE)
  if(ANDROID_NOEXECSTACK)
    set(ANDROID_DISABLE_NO_EXECUTE FALSE)
  else()
    set(ANDROID_DISABLE_NO_EXECUTE TRUE)
  endif()
endif()
if(DEFINED ANDROID_RELRO AND NOT DEFINED ANDROID_DISABLE_RELRO)
  if(ANDROID_RELRO)
    set(ANDROID_DISABLE_RELRO FALSE)
  else()
    set(ANDROID_DISABLE_RELRO TRUE)
  endif()
endif()
if(NDK_CCACHE AND NOT ANDROID_CCACHE)
  set(ANDROID_CCACHE "${NDK_CCACHE}")
endif()

# Default values for configurable variables.
if(NOT ANDROID_TOOLCHAIN)
  set(ANDROID_TOOLCHAIN gcc)
endif()
if(NOT ANDROID_ABI)
  set(ANDROID_ABI armeabi-v7a)
endif()
if(ANDROID_PLATFORM MATCHES "^android-([0-9]|1[0-3])$")
  set(ANDROID_PLATFORM android-14)
elseif(ANDROID_PLATFORM STREQUAL android-20)
  set(ANDROID_PLATFORM android-19)
elseif(ANDROID_PLATFORM STREQUAL android-25)
  set(ANDROID_PLATFORM android-24)
elseif(NOT ANDROID_PLATFORM)
  set(ANDROID_PLATFORM android-14)
endif()
string(REPLACE "android-" "" ANDROID_PLATFORM_LEVEL ${ANDROID_PLATFORM})
if(ANDROID_ABI MATCHES "64(-v8a)?$" AND ANDROID_PLATFORM_LEVEL LESS 21)
  set(ANDROID_PLATFORM android-21)
  set(ANDROID_PLATFORM_LEVEL 21)
endif()
if(NOT ANDROID_STL)
  set(ANDROID_STL gnustl_static)
endif()
if(NOT DEFINED ANDROID_PIE)
  if(ANDROID_PLATFORM_LEVEL LESS 16)
    set(ANDROID_PIE FALSE)
  else()
    set(ANDROID_PIE TRUE)
  endif()
endif()
if(NOT ANDROID_ARM_MODE)
  set(ANDROID_ARM_MODE thumb)
endif()

# Export configurable variables for the try_compile() command.
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  ANDROID_TOOLCHAIN
  ANDROID_ABI
  ANDROID_PLATFORM
  ANDROID_STL
  ANDROID_PIE
  ANDROID_CPP_FEATURES
  ANDROID_ALLOW_UNDEFINED_SYMBOLS
  ANDROID_ARM_MODE
  ANDROID_ARM_NEON
  ANDROID_DISABLE_NO_EXECUTE
  ANDROID_DISABLE_RELRO
  ANDROID_DISABLE_FORMAT_STRING_CHECKS
  ANDROID_CCACHE)

# Standard cross-compiling stuff.
set(ANDROID TRUE)
set(CMAKE_SYSTEM_NAME Android)

# Allow users to override these values in case they want more strict behaviors.
# For example, they may want to prevent the NDK's libz from being picked up so
# they can use their own.
# https://github.com/android-ndk/ndk/issues/517
if(NOT CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
endif()

if(NOT CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
endif()

if(NOT CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endif()

if(NOT CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()

# ABI.
set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
if(ANDROID_ABI MATCHES "^armeabi(-v7a)?$")
  set(ANDROID_SYSROOT_ABI arm)
  set(ANDROID_TOOLCHAIN_NAME arm-linux-androideabi)
  set(ANDROID_TOOLCHAIN_ROOT ${ANDROID_TOOLCHAIN_NAME})
  set(ANDROID_HEADER_TRIPLE arm-linux-androideabi)
  if(ANDROID_ABI STREQUAL armeabi)
    message(WARNING "armeabi is deprecated and will be removed in a future NDK "
                    "release.")
    set(CMAKE_SYSTEM_PROCESSOR armv5te)
    set(ANDROID_LLVM_TRIPLE armv5te-none-linux-androideabi)
  elseif(ANDROID_ABI STREQUAL armeabi-v7a)
    set(CMAKE_SYSTEM_PROCESSOR armv7-a)
    set(ANDROID_LLVM_TRIPLE armv7-none-linux-androideabi)
  endif()
elseif(ANDROID_ABI STREQUAL arm64-v8a)
  set(ANDROID_SYSROOT_ABI arm64)
  set(CMAKE_SYSTEM_PROCESSOR aarch64)
  set(ANDROID_TOOLCHAIN_NAME aarch64-linux-android)
  set(ANDROID_TOOLCHAIN_ROOT ${ANDROID_TOOLCHAIN_NAME})
  set(ANDROID_LLVM_TRIPLE aarch64-none-linux-android)
  set(ANDROID_HEADER_TRIPLE aarch64-linux-android)
elseif(ANDROID_ABI STREQUAL x86)
  set(ANDROID_SYSROOT_ABI x86)
  set(CMAKE_SYSTEM_PROCESSOR i686)
  set(ANDROID_TOOLCHAIN_NAME i686-linux-android)
  set(ANDROID_TOOLCHAIN_ROOT ${ANDROID_ABI})
  set(ANDROID_LLVM_TRIPLE i686-none-linux-android)
  set(ANDROID_HEADER_TRIPLE i686-linux-android)
elseif(ANDROID_ABI STREQUAL x86_64)
  set(ANDROID_SYSROOT_ABI x86_64)
  set(CMAKE_SYSTEM_PROCESSOR x86_64)
  set(ANDROID_TOOLCHAIN_NAME x86_64-linux-android)
  set(ANDROID_TOOLCHAIN_ROOT ${ANDROID_ABI})
  set(ANDROID_LLVM_TRIPLE x86_64-none-linux-android)
  set(ANDROID_HEADER_TRIPLE x86_64-linux-android)
elseif(ANDROID_ABI STREQUAL mips)
  message(WARNING "mips is deprecated and will be removed in a future NDK "
                  "release.")
  set(ANDROID_SYSROOT_ABI mips)
  set(CMAKE_SYSTEM_PROCESSOR mips)
  set(ANDROID_TOOLCHAIN_NAME mips64el-linux-android)
  set(ANDROID_TOOLCHAIN_ROOT ${ANDROID_TOOLCHAIN_NAME})
  set(ANDROID_LLVM_TRIPLE mipsel-none-linux-android)
  set(ANDROID_HEADER_TRIPLE mipsel-linux-android)
elseif(ANDROID_ABI STREQUAL mips64)
  message(WARNING "mips64 is deprecated and will be removed in a future NDK "
                  "release.")
  set(ANDROID_SYSROOT_ABI mips64)
  set(CMAKE_SYSTEM_PROCESSOR mips64)
  set(ANDROID_TOOLCHAIN_NAME mips64el-linux-android)
  set(ANDROID_TOOLCHAIN_ROOT ${ANDROID_TOOLCHAIN_NAME})
  set(ANDROID_LLVM_TRIPLE mips64el-none-linux-android)
  set(ANDROID_HEADER_TRIPLE mips64el-linux-android)
else()
  message(FATAL_ERROR "Invalid Android ABI: ${ANDROID_ABI}.")
endif()

set(ANDROID_COMPILER_FLAGS)
set(ANDROID_COMPILER_FLAGS_CXX)
set(ANDROID_COMPILER_FLAGS_DEBUG)
set(ANDROID_COMPILER_FLAGS_RELEASE)
set(ANDROID_LINKER_FLAGS)
set(ANDROID_LINKER_FLAGS_EXE)

# Don't re-export libgcc symbols in every binary.
list(APPEND ANDROID_LINKER_FLAGS -Wl,--exclude-libs,libgcc.a)
list(APPEND ANDROID_LINKER_FLAGS -Wl,--exclude-libs,libatomic.a)

# STL.
set(ANDROID_STL_STATIC_LIBRARIES)
set(ANDROID_STL_SHARED_LIBRARIES)
if(ANDROID_STL STREQUAL system)
  if(NOT "x${ANDROID_CPP_FEATURES}" STREQUAL "x")
    set(ANDROID_STL_STATIC_LIBRARIES supc++)
  endif()
elseif(ANDROID_STL STREQUAL stlport_static)
  set(ANDROID_STL_STATIC_LIBRARIES stlport_static)
elseif(ANDROID_STL STREQUAL stlport_shared)
  set(ANDROID_STL_SHARED_LIBRARIES stlport_shared)
elseif(ANDROID_STL STREQUAL gnustl_static)
  set(ANDROID_STL_STATIC_LIBRARIES gnustl_static)
elseif(ANDROID_STL STREQUAL gnustl_shared)
  set(ANDROID_STL_STATIC_LIBRARIES supc++)
  set(ANDROID_STL_SHARED_LIBRARIES gnustl_shared)
elseif(ANDROID_STL STREQUAL c++_static)
  set(ANDROID_STL_STATIC_LIBRARIES c++)
elseif(ANDROID_STL STREQUAL c++_shared)
  set(ANDROID_STL_SHARED_LIBRARIES c++)
elseif(ANDROID_STL STREQUAL none)
else()
  message(FATAL_ERROR "Invalid Android STL: ${ANDROID_STL}.")
endif()

# Behavior of CMAKE_SYSTEM_LIBRARY_PATH and CMAKE_LIBRARY_PATH are really weird
# when CMAKE_SYSROOT is set. The library path is appended to the sysroot even if
# the library path is an abspath. Using a relative path from the sysroot doesn't
# work either, because the relative path is abspath'd relative to the current
# CMakeLists.txt file before being appended :(
#
# We can try to get out of this problem by providing another root path for cmake
# to check. CMAKE_FIND_ROOT_PATH is intended for this purpose:
# https://cmake.org/cmake/help/v3.8/variable/CMAKE_FIND_ROOT_PATH.html
#
# In theory this should just be our sysroot, but since we don't have a single
# sysroot that is correct (there's only one set of headers, but multiple
# locations for libraries that need to be handled differently).  Some day we'll
# want to move all the libraries into ${ANDROID_NDK}/sysroot, but we'll need to
# make some fixes to Clang, various build systems, and possibly CMake itself to
# get that working.
list(APPEND CMAKE_FIND_ROOT_PATH "${ANDROID_NDK}")

# Sysroot.
set(CMAKE_SYSROOT "${ANDROID_NDK}/sysroot")

# CMake 3.9 tries to use CMAKE_SYSROOT_COMPILE before it gets set from
# CMAKE_SYSROOT, which leads to using the system's /usr/include. Set this
# manually.
# https://github.com/android-ndk/ndk/issues/467
set(CMAKE_SYSROOT_COMPILE "${CMAKE_SYSROOT}")

# The compiler driver doesn't check any arch specific include locations (though
# maybe we should add that). Architecture specific headers like asm/ and
# machine/ are installed to an arch-$ARCH subdirectory of the sysroot.
list(APPEND ANDROID_COMPILER_FLAGS
  "-isystem ${CMAKE_SYSROOT}/usr/include/${ANDROID_HEADER_TRIPLE}")
list(APPEND ANDROID_COMPILER_FLAGS
  "-D__ANDROID_API__=${ANDROID_PLATFORM_LEVEL}")

# We need different sysroots for linking and compiling, but cmake doesn't
# support that. Pass the sysroot flag manually when linking.
set(ANDROID_SYSTEM_LIBRARY_PATH
  "${ANDROID_NDK}/platforms/${ANDROID_PLATFORM}/arch-${ANDROID_SYSROOT_ABI}")
list(APPEND ANDROID_LINKER_FLAGS "--sysroot ${ANDROID_SYSTEM_LIBRARY_PATH}")

# find_library searches a handful of paths as described by
# https://cmake.org/cmake/help/v3.6/command/find_library.html.  Since libraries
# are per-API level and headers aren't, We don't have libraries in the
# CMAKE_SYSROOT. Set up CMAKE_SYSTEM_LIBRARY_PATH
# (https://cmake.org/cmake/help/v3.6/variable/CMAKE_SYSTEM_LIBRARY_PATH.html)
# instead.
#
# NB: The suffix is just lib here instead of dealing with lib64 because
# apparently CMake does some automatic rewriting of that? I've been testing by
# building my own CMake with a bunch of logging added, and that seems to be the
# case.
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH
  "${ANDROID_SYSTEM_LIBRARY_PATH}/usr/lib")

# Toolchain.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL Linux)
  set(ANDROID_HOST_TAG linux-x86_64)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL Darwin)
  set(ANDROID_HOST_TAG darwin-x86_64)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
  set(ANDROID_HOST_TAG windows-x86_64)
endif()
set(ANDROID_TOOLCHAIN_ROOT "${ANDROID_NDK}/toolchains/${ANDROID_TOOLCHAIN_ROOT}-4.9/prebuilt/${ANDROID_HOST_TAG}")
set(ANDROID_TOOLCHAIN_PREFIX "${ANDROID_TOOLCHAIN_ROOT}/bin/${ANDROID_TOOLCHAIN_NAME}-")
if(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
  set(ANDROID_TOOLCHAIN_SUFFIX .exe)
endif()

set(ANDROID_HOST_PREBUILTS "${ANDROID_NDK}/prebuilt/${ANDROID_HOST_TAG}")

if(ANDROID_TOOLCHAIN STREQUAL clang)
  set(ANDROID_LLVM_TOOLCHAIN_PREFIX "${ANDROID_NDK}/toolchains/llvm/prebuilt/${ANDROID_HOST_TAG}/bin/")
  set(ANDROID_C_COMPILER   "${ANDROID_LLVM_TOOLCHAIN_PREFIX}clang${ANDROID_TOOLCHAIN_SUFFIX}")
  set(ANDROID_CXX_COMPILER "${ANDROID_LLVM_TOOLCHAIN_PREFIX}clang++${ANDROID_TOOLCHAIN_SUFFIX}")
  set(ANDROID_ASM_COMPILER "${ANDROID_LLVM_TOOLCHAIN_PREFIX}clang${ANDROID_TOOLCHAIN_SUFFIX}")
  # Clang can fail to compile if CMake doesn't correctly supply the target and
  # external toolchain, but to do so, CMake needs to already know that the
  # compiler is clang. Tell CMake that the compiler is really clang, but don't
  # use CMakeForceCompiler, since we still want compile checks. We only want
  # to skip the compiler ID detection step.
  set(CMAKE_C_COMPILER_ID_RUN TRUE)
  set(CMAKE_CXX_COMPILER_ID_RUN TRUE)
  set(CMAKE_C_COMPILER_ID Clang)
  set(CMAKE_CXX_COMPILER_ID Clang)
  set(CMAKE_C_COMPILER_VERSION 3.8)
  set(CMAKE_CXX_COMPILER_VERSION 3.8)
  set(CMAKE_C_STANDARD_COMPUTED_DEFAULT 11)
  set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT 98)
  set(CMAKE_C_COMPILER_TARGET   ${ANDROID_LLVM_TRIPLE})
  set(CMAKE_CXX_COMPILER_TARGET ${ANDROID_LLVM_TRIPLE})
  set(CMAKE_ASM_COMPILER_TARGET ${ANDROID_LLVM_TRIPLE})
  set(CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN   "${ANDROID_TOOLCHAIN_ROOT}")
  set(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN "${ANDROID_TOOLCHAIN_ROOT}")
  set(CMAKE_ASM_COMPILER_EXTERNAL_TOOLCHAIN "${ANDROID_TOOLCHAIN_ROOT}")
  set(ANDROID_AR "${ANDROID_TOOLCHAIN_PREFIX}ar${ANDROID_TOOLCHAIN_SUFFIX}")
  set(ANDROID_RANLIB "${ANDROID_TOOLCHAIN_PREFIX}ranlib${ANDROID_TOOLCHAIN_SUFFIX}")
elseif(ANDROID_TOOLCHAIN STREQUAL gcc)
  set(ANDROID_C_COMPILER   "${ANDROID_TOOLCHAIN_PREFIX}gcc${ANDROID_TOOLCHAIN_SUFFIX}")
  set(ANDROID_CXX_COMPILER "${ANDROID_TOOLCHAIN_PREFIX}g++${ANDROID_TOOLCHAIN_SUFFIX}")
  set(ANDROID_ASM_COMPILER "${ANDROID_TOOLCHAIN_PREFIX}gcc${ANDROID_TOOLCHAIN_SUFFIX}")
  set(ANDROID_AR "${ANDROID_TOOLCHAIN_PREFIX}gcc-ar${ANDROID_TOOLCHAIN_SUFFIX}")
  set(ANDROID_RANLIB "${ANDROID_TOOLCHAIN_PREFIX}gcc-ranlib${ANDROID_TOOLCHAIN_SUFFIX}")
else()
  message(FATAL_ERROR "Invalid Android toolchain: ${ANDROID_TOOLCHAIN}.")
endif()

if(NOT IS_DIRECTORY "${ANDROID_NDK}/platforms/${ANDROID_PLATFORM}")
  message(FATAL_ERROR "Invalid Android platform: ${ANDROID_PLATFORM}.")
elseif(NOT IS_DIRECTORY "${CMAKE_SYSROOT}")
  message(FATAL_ERROR "Invalid Android sysroot: ${CMAKE_SYSROOT}.")
endif()

# Generic flags.
list(APPEND ANDROID_COMPILER_FLAGS
#  -g
  -DANDROID
  -ffunction-sections
  -funwind-tables
  -fstack-protector-strong
  -no-canonical-prefixes)
list(APPEND ANDROID_LINKER_FLAGS
  -Wl,--build-id
  -Wl,--warn-shared-textrel
  -Wl,--fatal-warnings)
list(APPEND ANDROID_LINKER_FLAGS_EXE
  -Wl,--gc-sections
  -Wl,-z,nocopyreloc)

# Debug and release flags.
list(APPEND ANDROID_COMPILER_FLAGS_DEBUG -O0)
if(ANDROID_ABI MATCHES "^armeabi")
  list(APPEND ANDROID_COMPILER_FLAGS_RELEASE -Os)
else()
  list(APPEND ANDROID_COMPILER_FLAGS_RELEASE -O2)
endif()
list(APPEND ANDROID_COMPILER_FLAGS_RELEASE -DNDEBUG)
if(ANDROID_TOOLCHAIN STREQUAL clang)
  list(APPEND ANDROID_COMPILER_FLAGS_DEBUG -fno-limit-debug-info)
endif()

# Toolchain and ABI specific flags.
if(ANDROID_ABI STREQUAL armeabi)
  list(APPEND ANDROID_COMPILER_FLAGS
    -march=armv5te
    -mtune=xscale
    -msoft-float)
endif()
if(ANDROID_ABI STREQUAL armeabi-v7a)
  list(APPEND ANDROID_COMPILER_FLAGS
    -march=armv7-a
    -mfloat-abi=softfp
    -mfpu=vfpv3-d16)
  list(APPEND ANDROID_LINKER_FLAGS
    -Wl,--fix-cortex-a8)
endif()
if(ANDROID_ABI STREQUAL mips)
  list(APPEND ANDROID_COMPILER_FLAGS
    -mips32)
endif()
if(ANDROID_ABI STREQUAL "mips64" AND ANDROID_TOOLCHAIN STREQUAL clang)
  list(APPEND ANDROID_COMPILER_FLAGS "-fintegrated-as")
endif()
if(ANDROID_ABI MATCHES "^armeabi" AND ANDROID_TOOLCHAIN STREQUAL clang)
  # Disable integrated-as for better compatibility.
  list(APPEND ANDROID_COMPILER_FLAGS
    -fno-integrated-as)
endif()
if(ANDROID_ABI STREQUAL mips AND ANDROID_TOOLCHAIN STREQUAL clang)
  # Help clang use mips64el multilib GCC
  list(APPEND ANDROID_LINKER_FLAGS
    "\"-L${ANDROID_TOOLCHAIN_ROOT}/lib/gcc/${ANDROID_TOOLCHAIN_NAME}/4.9.x/32/mips-r1\"")
endif()
if(ANDROID_ABI STREQUAL x86)
  # http://b.android.com/222239
  # http://b.android.com/220159 (internal http://b/31809417)
  # x86 devices have stack alignment issues.
  list(APPEND ANDROID_COMPILER_FLAGS -mstackrealign)
endif()

# STL specific flags.
if(ANDROID_STL STREQUAL system)
  set(ANDROID_STL_PREFIX gnu-libstdc++/4.9)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    "${ANDROID_NDK}/sources/cxx-stl/system/include")
elseif(ANDROID_STL MATCHES "^stlport_")
  set(ANDROID_STL_PREFIX stlport)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    "${ANDROID_NDK}/sources/cxx-stl/${ANDROID_STL_PREFIX}/stlport"
    "${ANDROID_NDK}/sources/cxx-stl/gabi++/include")
elseif(ANDROID_STL MATCHES "^gnustl_")
  set(ANDROID_STL_PREFIX gnu-libstdc++/4.9)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    "${ANDROID_NDK}/sources/cxx-stl/${ANDROID_STL_PREFIX}/include"
    "${ANDROID_NDK}/sources/cxx-stl/${ANDROID_STL_PREFIX}/libs/${ANDROID_ABI}/include"
    "${ANDROID_NDK}/sources/cxx-stl/${ANDROID_STL_PREFIX}/include/backward")
elseif(ANDROID_STL MATCHES "^c\\+\\+_")
  set(ANDROID_STL_PREFIX llvm-libc++)
  if(ANDROID_ABI MATCHES "^armeabi")
    list(APPEND ANDROID_LINKER_FLAGS -Wl,--exclude-libs,libunwind.a)
  endif()
  list(APPEND ANDROID_COMPILER_FLAGS_CXX
    -std=c++11)
  if(ANDROID_TOOLCHAIN STREQUAL gcc)
    list(APPEND ANDROID_COMPILER_FLAGS_CXX
      -fno-strict-aliasing)
  endif()

  # Add the libc++ lib directory to the path so the linker scripts can pick up
  # the extra libraries.
  list(APPEND ANDROID_LINKER_FLAGS
    "-L${ANDROID_NDK}/sources/cxx-stl/${ANDROID_STL_PREFIX}/libs/${ANDROID_ABI}")

  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    "${ANDROID_NDK}/sources/cxx-stl/${ANDROID_STL_PREFIX}/include"
    "${ANDROID_NDK}/sources/android/support/include"
    "${ANDROID_NDK}/sources/cxx-stl/${ANDROID_STL_PREFIX}abi/include")
endif()
set(ANDROID_CXX_STANDARD_LIBRARIES)
foreach(library ${ANDROID_STL_STATIC_LIBRARIES})
  list(APPEND ANDROID_CXX_STANDARD_LIBRARIES
    "${ANDROID_NDK}/sources/cxx-stl/${ANDROID_STL_PREFIX}/libs/${ANDROID_ABI}/lib${library}.a")
endforeach()
foreach(library ${ANDROID_STL_SHARED_LIBRARIES})
  list(APPEND ANDROID_CXX_STANDARD_LIBRARIES
    "${ANDROID_NDK}/sources/cxx-stl/${ANDROID_STL_PREFIX}/libs/${ANDROID_ABI}/lib${library}.so")
endforeach()
set(CMAKE_C_STANDARD_LIBRARIES_INIT "-latomic -lm")
set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_C_STANDARD_LIBRARIES_INIT}")
if(ANDROID_CXX_STANDARD_LIBRARIES)
  string(REPLACE ";" "\" \"" ANDROID_CXX_STANDARD_LIBRARIES "\"${ANDROID_CXX_STANDARD_LIBRARIES}\"")
  set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_CXX_STANDARD_LIBRARIES}")
endif()

# Configuration specific flags.
if(ANDROID_PIE)
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
  list(APPEND ANDROID_LINKER_FLAGS_EXE
    -pie
    -fPIE)
endif()
if(ANDROID_CPP_FEATURES)
  separate_arguments(ANDROID_CPP_FEATURES)
  foreach(feature ${ANDROID_CPP_FEATURES})
    if(NOT ${feature} MATCHES "^(rtti|exceptions)$")
      message(FATAL_ERROR "Invalid Android C++ feature: ${feature}.")
    endif()
    list(APPEND ANDROID_COMPILER_FLAGS_CXX
      -f${feature})
  endforeach()
  string(REPLACE ";" " " ANDROID_CPP_FEATURES "${ANDROID_CPP_FEATURES}")
endif()
if(NOT ANDROID_ALLOW_UNDEFINED_SYMBOLS)
  list(APPEND ANDROID_LINKER_FLAGS
    -Wl,--no-undefined)
endif()
if(ANDROID_ABI MATCHES "armeabi")
  if(ANDROID_ARM_MODE STREQUAL thumb)
    list(APPEND ANDROID_COMPILER_FLAGS
      -mthumb)
  elseif(ANDROID_ARM_MODE STREQUAL arm)
    list(APPEND ANDROID_COMPILER_FLAGS
      -marm)
  else()
    message(FATAL_ERROR "Invalid Android ARM mode: ${ANDROID_ARM_MODE}.")
  endif()
  if(ANDROID_ABI STREQUAL armeabi-v7a AND ANDROID_ARM_NEON)
    list(APPEND ANDROID_COMPILER_FLAGS
      -mfpu=neon)
  endif()
endif()
if(ANDROID_DISABLE_NO_EXECUTE)
  list(APPEND ANDROID_COMPILER_FLAGS
    -Wa,--execstack)
  list(APPEND ANDROID_LINKER_FLAGS
    -Wl,-z,execstack)
else()
  list(APPEND ANDROID_COMPILER_FLAGS
    -Wa,--noexecstack)
  list(APPEND ANDROID_LINKER_FLAGS
    -Wl,-z,noexecstack)
endif()
if(ANDROID_TOOLCHAIN STREQUAL clang)
  # CMake automatically forwards all compiler flags to the linker,
  # and clang doesn't like having -Wa flags being used for linking.
  # To prevent CMake from doing this would require meddling with
  # the CMAKE_<LANG>_COMPILE_OBJECT rules, which would get quite messy.
  list(APPEND ANDROID_LINKER_FLAGS
    -Qunused-arguments)
endif()
if(ANDROID_DISABLE_RELRO)
  list(APPEND ANDROID_LINKER_FLAGS
    -Wl,-z,norelro -Wl,-z,lazy)
else()
  list(APPEND ANDROID_LINKER_FLAGS
    -Wl,-z,relro -Wl,-z,now)
endif()
if(ANDROID_DISABLE_FORMAT_STRING_CHECKS)
  list(APPEND ANDROID_COMPILER_FLAGS
    -Wno-error=format-security)
else()
  list(APPEND ANDROID_COMPILER_FLAGS
    -Wformat -Werror=format-security)
endif()

# Convert these lists into strings.
string(REPLACE ";" " " ANDROID_COMPILER_FLAGS         "${ANDROID_COMPILER_FLAGS}")
string(REPLACE ";" " " ANDROID_COMPILER_FLAGS_CXX     "${ANDROID_COMPILER_FLAGS_CXX}")
string(REPLACE ";" " " ANDROID_COMPILER_FLAGS_DEBUG   "${ANDROID_COMPILER_FLAGS_DEBUG}")
string(REPLACE ";" " " ANDROID_COMPILER_FLAGS_RELEASE "${ANDROID_COMPILER_FLAGS_RELEASE}")
string(REPLACE ";" " " ANDROID_LINKER_FLAGS           "${ANDROID_LINKER_FLAGS}")
string(REPLACE ";" " " ANDROID_LINKER_FLAGS_EXE       "${ANDROID_LINKER_FLAGS_EXE}")

if(ANDROID_CCACHE)
  set(CMAKE_C_COMPILER_LAUNCHER   "${ANDROID_CCACHE}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${ANDROID_CCACHE}")
endif()
set(CMAKE_C_COMPILER        "${ANDROID_C_COMPILER}")
set(CMAKE_CXX_COMPILER      "${ANDROID_CXX_COMPILER}")
set(CMAKE_AR                "${ANDROID_AR}" CACHE FILEPATH "Archiver")
set(CMAKE_RANLIB            "${ANDROID_RANLIB}" CACHE FILEPATH "Ranlib")
set(_CMAKE_TOOLCHAIN_PREFIX "${ANDROID_TOOLCHAIN_PREFIX}")

if(ANDROID_ABI STREQUAL "x86" OR ANDROID_ABI STREQUAL "x86_64")
  set(CMAKE_ASM_NASM_COMPILER
    "${ANDROID_HOST_PREBUILTS}/bin/yasm${ANDROID_TOOLCHAIN_SUFFIX}")
  set(CMAKE_ASM_NASM_COMPILER_ARG1 "-DELF")
endif()

# Set or retrieve the cached flags.
# This is necessary in case the user sets/changes flags in subsequent
# configures. If we included the Android flags in here, they would get
# overwritten.
set(CMAKE_C_FLAGS ""
  CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_CXX_FLAGS ""
  CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_ASM_FLAGS ""
  CACHE STRING "Flags used by the compiler during all build types.")
set(CMAKE_C_FLAGS_DEBUG ""
  CACHE STRING "Flags used by the compiler during debug builds.")
set(CMAKE_CXX_FLAGS_DEBUG ""
  CACHE STRING "Flags used by the compiler during debug builds.")
set(CMAKE_ASM_FLAGS_DEBUG ""
  CACHE STRING "Flags used by the compiler during debug builds.")
set(CMAKE_C_FLAGS_RELEASE ""
  CACHE STRING "Flags used by the compiler during release builds.")
set(CMAKE_CXX_FLAGS_RELEASE ""
  CACHE STRING "Flags used by the compiler during release builds.")
set(CMAKE_ASM_FLAGS_RELEASE ""
  CACHE STRING "Flags used by the compiler during release builds.")
set(CMAKE_MODULE_LINKER_FLAGS ""
  CACHE STRING "Flags used by the linker during the creation of modules.")
set(CMAKE_SHARED_LINKER_FLAGS ""
  CACHE STRING "Flags used by the linker during the creation of dll's.")
set(CMAKE_EXE_LINKER_FLAGS ""
  CACHE STRING "Flags used by the linker.")

set(CMAKE_C_FLAGS             "${ANDROID_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS           "${ANDROID_COMPILER_FLAGS} ${ANDROID_COMPILER_FLAGS_CXX} ${CMAKE_CXX_FLAGS}")
set(CMAKE_ASM_FLAGS           "${ANDROID_COMPILER_FLAGS} ${CMAKE_ASM_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG       "${ANDROID_COMPILER_FLAGS_DEBUG} ${CMAKE_C_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_DEBUG     "${ANDROID_COMPILER_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_ASM_FLAGS_DEBUG     "${ANDROID_COMPILER_FLAGS_DEBUG} ${CMAKE_ASM_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_RELEASE     "${ANDROID_COMPILER_FLAGS_RELEASE} ${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELEASE   "${ANDROID_COMPILER_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_ASM_FLAGS_RELEASE   "${ANDROID_COMPILER_FLAGS_RELEASE} ${CMAKE_ASM_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS "${ANDROID_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${ANDROID_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS    "${ANDROID_LINKER_FLAGS} ${ANDROID_LINKER_FLAGS_EXE} ${CMAKE_EXE_LINKER_FLAGS}")

# Compatibility for read-only variables.
# Read-only variables for compatibility with the other toolchain file.
# We'll keep these around for the existing projects that still use them.
# TODO: All of the variables here have equivalents in our standard set of
# configurable variables, so we can remove these once most of our users migrate
# to those variables.
set(ANDROID_NATIVE_API_LEVEL ${ANDROID_PLATFORM_LEVEL})
if(ANDROID_ALLOW_UNDEFINED_SYMBOLS)
  set(ANDROID_SO_UNDEFINED TRUE)
else()
  set(ANDROID_NO_UNDEFINED TRUE)
endif()
set(ANDROID_FUNCTION_LEVEL_LINKING TRUE)
set(ANDROID_GOLD_LINKER TRUE)
if(NOT ANDROID_DISABLE_NO_EXECUTE)
  set(ANDROID_NOEXECSTACK TRUE)
endif()
if(NOT ANDROID_DISABLE_RELRO)
  set(ANDROID_RELRO TRUE)
endif()
if(ANDROID_ARM_MODE STREQUAL arm)
  set(ANDROID_FORCE_ARM_BUILD TRUE)
endif()
if(ANDROID_CPP_FEATURES MATCHES "rtti"
    AND ANDROID_CPP_FEATURES MATCHES "exceptions")
  set(ANDROID_STL_FORCE_FEATURES TRUE)
endif()
if(ANDROID_CCACHE)
  set(NDK_CCACHE "${ANDROID_CCACHE}")
endif()
if(ANDROID_TOOLCHAIN STREQUAL clang)
  set(ANDROID_TOOLCHAIN_NAME ${ANDROID_TOOLCHAIN_NAME}-clang)
else()
  set(ANDROID_TOOLCHAIN_NAME ${ANDROID_TOOLCHAIN_NAME}-4.9)
endif()
set(ANDROID_NDK_HOST_X64 TRUE)
set(ANDROID_NDK_LAYOUT RELEASE)
if(ANDROID_ABI STREQUAL armeabi)
  set(ARMEABI TRUE)
elseif(ANDROID_ABI STREQUAL armeabi-v7a)
  set(ARMEABI_V7A TRUE)
  if(ANDROID_ARM_NEON)
    set(NEON TRUE)
  endif()
elseif(ANDROID_ABI STREQUAL arm64-v8a)
  set(ARM64_V8A TRUE)
elseif(ANDROID_ABI STREQUAL x86)
  set(X86 TRUE)
elseif(ANDROID_ABI STREQUAL x86_64)
  set(X86_64 TRUE)
elseif(ANDROID_ABI STREQUAL mips)
  set(MIPS TRUE)
elseif(ANDROID_ABI STREQUAL mips64)
  set(MIPS64 TRUE)
endif()
set(ANDROID_NDK_HOST_SYSTEM_NAME ${ANDROID_HOST_TAG})
set(ANDROID_NDK_ABI_NAME ${ANDROID_ABI})
set(ANDROID_NDK_RELEASE r${ANDROID_NDK_REVISION})
set(ANDROID_ARCH_NAME ${ANDROID_SYSROOT_ABI})
set(ANDROID_SYSROOT "${CMAKE_SYSROOT}")
set(TOOL_OS_SUFFIX ${ANDROID_TOOLCHAIN_SUFFIX})
if(ANDROID_TOOLCHAIN STREQUAL clang)
  set(ANDROID_COMPILER_IS_CLANG TRUE)
endif()

# CMake 3.7+ compatibility.
if (CMAKE_VERSION VERSION_GREATER 3.7.0)
  set(CMAKE_ANDROID_NDK ${ANDROID_NDK})

  if(ANDROID_TOOLCHAIN STREQUAL gcc)
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION 4.9)
  else()
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)
  endif()

  set(CMAKE_ANDROID_STL_TYPE ${ANDROID_STL})

  if(ANDROID_ABI MATCHES "^armeabi(-v7a)?$")
    set(CMAKE_ANDROID_ARM_NEON ${ANDROID_ARM_NEON})
    set(CMAKE_ANDROID_ARM_MODE ${ANDROID_ARM_MODE})
  endif()
endif()
