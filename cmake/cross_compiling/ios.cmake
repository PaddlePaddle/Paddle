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

# This is a toolchain file for cross-compiling for iOS, and the
# configuration largely refers to public toolchain file:
#    https://raw.githubusercontent.com/leetal/ios-cmake/master/ios.toolchain.cmake
# and
#    https://github.com/cristeab/ios-cmake
#
# Supports options:
# IOS_PLATFORM = OS (default) or SIMULATOR
#   This decides if SDKS will be selected from the iPhoneOS.platform or iPhoneSimulator.platform folders
#   OS - the default, used to build for iPhone and iPad physical devices, which have an arm arch.
#   SIMULATOR - used to build for the Simulator platforms, which have an x86 arch.
# IOS_ARCH
#   The archectures wanted to support, such "arm64", "armv7;arm64"
# IOS_DEPLOYMENT_TARGET
#   The minimum iOS deployment version, such as "7.0"
# IOS_ENABLE_BITCODE = ON (default) or OFF
# IOS_USE_VECLIB_FOR_BLAS = OFF (default) or ON
# IOS_DEVELOPER_ROOT = automatic(default) or /path/to/platform/Developer folder
#   By default this location is automatcially chosen based on the IOS_PLATFORM value above.
#   If set manually, it will override the default location and force the user of a particular Developer Platform
# IOS_SDK_ROOT = automatic(default) or /path/to/platform/Developer/SDKs/SDK folder
#   By default this location is automatcially chosen based on the IOS_DEVELOPER_ROOT value.
#   In this case it will always be the most up-to-date SDK found in the IOS_DEVELOPER_ROOT path.
#   If set manually, this will force the use of a specific SDK version

# Macros:
# set_xcode_property (TARGET XCODE_PROPERTY XCODE_VALUE)
#  A convenience macro for setting xcode specific properties on targets
#  example: set_xcode_property (myioslib IPHONEOS_DEPLOYMENT_TARGET "3.1")
# find_host_package (PROGRAM ARGS)
#  A macro used to find executable programs on the host system, not within the iOS environment.
#  Thanks to the android-cmake project for providing the command

if(NOT IOS)
  return()
endif()

set(CMAKE_SYSTEM_NAME Darwin)

# Get the Xcode version being used.
execute_process(COMMAND xcodebuild -version
                OUTPUT_VARIABLE XCODE_VERSION
                RESULT_VARIABLE XCODE_VERSION_RESULT
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ${XCODE_VERSION_RESULT})
  string(REGEX MATCH "Xcode [0-9\\.]+" XCODE_VERSION "${XCODE_VERSION}")
  string(REGEX REPLACE "Xcode ([0-9\\.]+)" "\\1" XCODE_VERSION "${XCODE_VERSION}")
  message(STATUS "Building with Xcode version: ${XCODE_VERSION}")
else()
  message(FATAL_ERROR "Cannot execute xcodebuild, please check whether xcode is installed.")
endif()

# Required as of cmake 2.8.10
set(CMAKE_OSX_DEPLOYMENT_TARGET "" CACHE STRING "Force unset of the deployment target for iOS" FORCE)

# Setup iOS platform unless specified manually with IOS_PLATFORM
if(NOT DEFINED IOS_PLATFORM)
  set(IOS_PLATFORM "OS")
endif()
set(IOS_PLATFORM ${IOS_PLATFORM} CACHE STRING "Type of iOS Platform")

# Set the architecture for iOS
if(NOT DEFINED IOS_ARCH)
  if(IOS_PLATFORM STREQUAL "OS")
    set(IOS_ARCH "armv7;armv7s;arm64")
  elseif(IOS_PLATFORM STREQUAL "SIMULATOR")
    set(IOS_ARCH "i386;x86_64")
  endif()
endif()
set(CMAKE_OSX_ARCHITECTURES ${IOS_ARCH} CACHE string  "Build architecture for iOS")

# Specify minimum iOS deployment version
if(NOT DEFINED IOS_DEPLOYMENT_TARGET)
  set(IOS_DEPLOYMENT_TARGET "7.0")
endif()
set(IOS_DEPLOYMENT_TARGET ${IOS_DEPLOYMENT_TARGET} CACHE STRING "Minimum iOS version")

# Whether to enable bitcode
if(NOT DEFINED IOS_ENABLE_BITCODE)
  set(IOS_ENABLE_BITCODE ON)
endif()
set(IOS_ENABLE_BITCODE ${IOS_ENABLE_BITCODE} CACHE BOOL "Whether to enable bitcode")

if(NOT DEFINED IOS_USE_VECLIB_FOR_BLAS)
  set(IOS_USE_VECLIB_FOR_BLAS OFF)
endif()
set(IOS_USE_VECLIB_FOR_BLAS ${IOS_UES_VECLIB_FOR_BLAS} CACHE BOOL "Whether to use veclib")

# Check the platform selection and setup for developer root
if(${IOS_PLATFORM} STREQUAL "OS")
  set(IOS_PLATFORM_LOCATION "iPhoneOS.platform")
  set(XCODE_IOS_PLATFORM iphoneos)

  # This causes the installers to properly locate the output libraries
  set(CMAKE_XCODE_EFFECTIVE_PLATFORMS "-iphoneos")
elseif(${IOS_PLATFORM} STREQUAL "SIMULATOR")
  set(IOS_PLATFORM_LOCATION "iPhoneSimulator.platform")
  set(XCODE_IOS_PLATFORM iphonesimulator)

  # This causes the installers to properly locate the output libraries
  set(CMAKE_XCODE_EFFECTIVE_PLATFORMS "-iphonesimulator")
elseif(${IOS_PLATFORM} STREQUAL "WATCHOS")
  set(IOS_PLATFORM_LOCATION "WatchOS.platform")
  set(XCODE_IOS_PLATFORM watchos)

  # This causes the installers to properly locate the output libraries
  set(CMAKE_XCODE_EFFECTIVE_PLATFORMS "-watchos")
else(${IOS_PLATFORM} STREQUAL "OS")
  message(FATAL_ERROR "Unsupported IOS_PLATFORM value selected. Please set to\n"
          "\t OS, SIMULATOR, or WATCHOS.")
endif()

# Check iOS developer toolchain
if(NOT DEFINED IOS_DEVELOPER_ROOT)
  # Setup iOS developer location
  execute_process(COMMAND xcode-select -print-path
                  OUTPUT_VARIABLE XCODE_DEVELOPER_DIR
                  RESULT_VARIABLE XCODE_DEVELOPER_DIR_RESULT
                  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  # Xcode 4.3 changed the installation location, choose the most recent one available
  if(${XCODE_VERSION} VERSION_LESS "4.3.0")
    set(IOS_DEVELOPER_ROOT "/Developer/Platforms/${IOS_PLATFORM_LOCATION}/Developer")
  else()
    set(IOS_DEVELOPER_ROOT "${XCODE_DEVELOPER_DIR}/Platforms/${IOS_PLATFORM_LOCATION}/Developer")
  endif()
endif()
if(EXISTS ${IOS_DEVELOPER_ROOT})
  set(IOS_DEVELOPER_ROOT ${IOS_DEVELOPER_ROOT} CACHE PATH "Location of iOS Platform")
else()
  message(FATAL_ERROR "Invalid IOS_DEVELOPER_ROOT: ${IOS_DEVELOPER_ROOT} does not exist.")
endif()

# Check iOS SDK
if(NOT DEFINED IOS_SDK_ROOT)
  # Find and use the most recent iOS sdk
  file(GLOB IOS_SDK_LISTS "${IOS_DEVELOPER_ROOT}/SDKs/*")
  if(IOS_SDK_LISTS)
    list(SORT IOS_SDK_LISTS)
    list(REVERSE IOS_SDK_LISTS)
    list(GET IOS_SDK_LISTS 0 IOS_SDK_ROOT)
  else(IOS_SDK_LISTS)
    message(FATAL_ERROR "No iOS SDK's found in default search path ${IOS_DEVELOPER_ROOT}."
            " Please manually set IOS_SDK_ROOT or install the iOS SDK.")
  endif(IOS_SDK_LISTS)
endif()
if(EXISTS ${IOS_SDK_ROOT})
  set(IOS_SDK_ROOT ${IOS_SDK_ROOT} CACHE PATH "Location of the selected iOS SDK")
  message(STATUS "iOS toolchain: ${IOS_SDK_ROOT}")
else()
  message(FATAL_ERROR "Invalid IOS_SDK_ROOT: ${IOS_SDK_ROOT} does not exist.")
endif()

# Set the sysroot default to the most recent SDK
set(CMAKE_OSX_SYSROOT ${IOS_SDK_ROOT} CACHE PATH "Sysroot used for iOS support")

# Get version of iOS SDK
execute_process(COMMAND xcodebuild -sdk ${CMAKE_OSX_SYSROOT} -version SDKVersion
                OUTPUT_VARIABLE IOS_SDK_VERSION
                RESULT_VARIABLE IOS_SDK_VERSION_RESULT
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if(${IOS_SDK_VERSION_RESULT})
  string(REGEX MATCH "(([0-9]+)\\.)+([0-9]+)" IOS_SDK_VERSION "${IOS_SDK_ROOT}")
endif()
if(NOT IOS_SDK_VERSION)
  message(WARNING "Cannot get SDK's version.")
  set(IOS_SDK_VERSION 1)
endif()
set(CMAKE_SYSTEM_VERSION ${IOS_SDK_VERSION})

# Find the C & C++ compilers for the specified SDK.
if(NOT CMAKE_C_COMPILER)
  # Default to use clang
  execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find clang
                  OUTPUT_VARIABLE IOS_C_COMPILER
                  RESULT_VARIABLE IOS_C_COMPILER_RESULT
                  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(${IOS_C_COMPILER_RESULT})
    get_filename_component(IOS_C_COMPILER clang PROGRAM)
  endif()
else(NOT CMAKE_C_COMPILER)
  # User can set it in cmake command
  get_filename_component(IOS_C_COMPILER ${CMAKE_C_COMPILER} PROGRAM)
endif(NOT CMAKE_C_COMPILER)
if(NOT EXISTS ${IOS_C_COMPILER})
  message(FATAL_ERROR "Cannot find C compiler: ${IOS_C_COMPILER}")
endif()

if(NOT CMAKE_CXX_COMPILER)
  # Default to use clang++
  execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find clang++
                  OUTPUT_VARIABLE IOS_CXX_COMPILER
                  RESULT_VARIABLE IOS_CXX_COMPILER_RESULT
                  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(${IOS_CXX_COMPILER_RESULT})
    get_filename_component(IOS_CXX_COMPILER clang++ PROGRAM)
  endif()
else(NOT CMAKE_CXX_COMPILER)
  # User can set it in cmake command
  get_filename_component(IOS_CXX_COMPILER ${CMAKE_CXX_COMPILER} PROGRAM)
endif(NOT CMAKE_CXX_COMPILER)
if(NOT EXISTS ${IOS_CXX_COMPILER})
  message(FATAL_ERROR "Cannot find CXX compiler: ${IOS_CXX_COMPILER}")
endif()

set(CMAKE_C_COMPILER ${IOS_C_COMPILER} CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER ${IOS_CXX_COMPILER} CACHE PATH "CXX compiler" FORCE)

set(CMAKE_C_OSX_COMPATIBILITY_VERSION_FLAG "-compatibility_version ")
set(CMAKE_C_OSX_CURRENT_VERSION_FLAG "-current_version ")
set(CMAKE_CXX_OSX_COMPATIBILITY_VERSION_FLAG "${CMAKE_C_OSX_COMPATIBILITY_VERSION_FLAG}")
set(CMAKE_CXX_OSX_CURRENT_VERSION_FLAG "${CMAKE_C_OSX_CURRENT_VERSION_FLAG}")

# Set iOS specific C/C++ flags
if(IOS_PLATFORM STREQUAL "OS")
  if(XCODE_VERSION VERSION_LESS "7.0")
    set(XCODE_IOS_PLATFORM_VERSION_FLAGS "-mios-version-min=${IOS_DEPLOYMENT_TARGET}")
  else()
    # Xcode 7.0+ uses flags we can build directly from XCODE_IOS_PLATFORM.
    set(XCODE_IOS_PLATFORM_VERSION_FLAGS "-m${XCODE_IOS_PLATFORM}-version-min=${IOS_DEPLOYMENT_TARGET}")
  endif()
else()
  set(XCODE_IOS_FLATFORM_VERSION_FLAGS "-mios-simulator-version-min=${IOS_DEPLOYMENT_TARGET}")
endif()

if(IOS_ENABLE_BITCODE)
  set(XCODE_IOS_BITCODE_FLAGS "${IOS_COMPILER_FLAGS} -fembed-bitcode")
else()
  set(XCODE_IOS_BITCODE_FLAGS "")
endif()

set(IOS_COMPILER_FLAGS "${XCODE_IOS_PLATFORM_VERSION_FLAGS} ${XCODE_IOS_BITCODE_FLAGS}")

# Hidden visibilty is required for cxx on iOS 
set(CMAKE_C_FLAGS "${IOS_COMPILER_FLAGS} ${CMAKE_C_FLAGS}" CACHE STRING "C flags")
set(CMAKE_CXX_FLAGS "${IOS_COMPILER_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden ${CMAKE_CXX_FLAGS}" CACHE STRING "CXX flags")

set(IOS_LINK_FLAGS "${XCODE_IOS_PLATFORM_VERSION_FLAGS} -Wl,-search_paths_first")

if(IOS_USE_VECLIB_FOR_BLAS)
  # Find vecLib for iOS
  set(VECLIB_SEARCH_DIRS
      ${IOS_SDK_ROOT}/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks
      ${IOS_SDK_ROOT}/System/Library/Frameworks/Accelerate.framework/Frameworks
      )
  find_path(VECLIB_INC_DIR vecLib.h PATHS ${VECLIB_SEARCH_DIRS}/vecLib.framework/Headers)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(vecLib DEFAULT_MSG VECLIB_INC_DIR)

  if(VECLIB_FOUND)
    if(VECLIB_INC_DIR MATCHES "^/System/Library/Frameworks/vecLib.framework.*")
      set(IOS_LINK_FLAGS ${IOS_LINK_FLAGS} -lcblas "-framework vecLib")
      message(STATUS "Found standalone vecLib.framework")
    else()
      set(IOS_LINK_FLAGS ${IOS_LINK_FLAGS} -lcblas "-framework Accelerate")
      message(STATUS "Found vecLib as part of Accelerate.framework")
    endif()

  endif()
endif()

set(CMAKE_C_LINK_FLAGS "${IOS_LINK_FLAGS} ${CMAKE_C_LINK_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS "${IOS_LINK_FLAGS} ${CMAKE_CXX_LINK_FLAGS}")

set(CMAKE_PLATFORM_HAS_INSTALLNAME 1)
if(NOT IOS_ENABLE_BITCODE)
  set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-dynamiclib -headerpad_max_install_names")
  set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS "-bundle -headerpad_max_install_names")
else()
  set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-dynamiclib")
  set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS "-bundle")
endif()
set(CMAKE_SHARED_MODULE_LOADER_C_FLAG "-Wl,-bundle_loader,")
set(CMAKE_SHARED_MODULE_LOADER_CXX_FLAG "-Wl,-bundle_loader,")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib" ".so" ".a")

# hack: if a new cmake (which uses CMAKE_INSTALL_NAME_TOOL) runs on an old build tree
# (where install_name_tool was hardcoded) and where CMAKE_INSTALL_NAME_TOOL isn't in the cache
# and still cmake didn't fail in CMakeFindBinUtils.cmake (because it isn't rerun)
# hardcode CMAKE_INSTALL_NAME_TOOL here to install_name_tool, so it behaves as it did before, Alex
if(NOT DEFINED CMAKE_INSTALL_NAME_TOOL)
  find_program(CMAKE_INSTALL_NAME_TOOL install_name_tool)
endif()

# Set the find root to the iOS developer roots and to user defined paths
set(CMAKE_FIND_ROOT_PATH ${IOS_DEVELOPER_ROOT} ${IOS_SDK_ROOT} ${CMAKE_PREFIX_PATH}
    CACHE string  "iOS find search path root")

# default to searching for frameworks first
set(CMAKE_FIND_FRAMEWORK FIRST)

# set up the default search directories for frameworks
set(CMAKE_SYSTEM_FRAMEWORK_PATH
    ${IOS_SDK_ROOT}/System/Library/Frameworks
    ${IOS_SDK_ROOT}/System/Library/PrivateFrameworks
    ${IOS_SDK_ROOT}/Developer/Library/Frameworks
    )

# only search the iOS sdks, not the remainder of the host filesystem
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

message(STATUS "iOS: Targeting iOS '${CMAKE_SYSTEM_VERSION}', "
        "building for '${IOS_PLATFORM}' platform, with architecture '${CMAKE_OSX_ARCHITECTURES}'")
message(STATUS "System CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "System CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# Used in ExternalProject command
string(REPLACE ";" "\\$<SEMICOLON>" EXTERNAL_IOS_ARCHITECTURES "${CMAKE_OSX_ARCHITECTURES}")
set(EXTERNAL_OPTIONAL_ARGS
    -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
    -DCMAKE_OSX_ARCHITECTURES=${EXTERNAL_IOS_ARCHITECTURES})

# This little macro lets you set any XCode specific property
macro(set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE)
  set_property (TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY} ${XCODE_VALUE})
endmacro(set_xcode_property)

# This macro lets you find executable programs on the host system
macro(find_host_package)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
  set(IOS FALSE)

  find_package(${ARGN})

  set(IOS TRUE)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endmacro(find_host_package)
