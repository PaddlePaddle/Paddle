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

include(ExternalProject)

message(STATUS "third: ${CINN_THIRD_PARTY_PATH}")
set(GFLAGS_SOURCES_DIR ${CINN_THIRD_PARTY_PATH}/gflags)
set(GFLAGS_INSTALL_DIR ${CINN_THIRD_PARTY_PATH}/install/gflags)
set(GFLAGS_INCLUDE_DIR
    "${GFLAGS_INSTALL_DIR}/include"
    CACHE PATH "gflags include directory." FORCE)
if(WIN32)
  set(GFLAGS_LIBRARIES
      "${GFLAGS_INSTALL_DIR}/lib/libgflags.lib"
      CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
else(WIN32)
  set(GFLAGS_LIBRARIES
      "${GFLAGS_INSTALL_DIR}/lib/libgflags.a"
      CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
endif(WIN32)

include_directories(${GFLAGS_INCLUDE_DIR})

set(OPTIONAL_ARGS
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
    "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
    "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
    "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
    "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")

if(ANDROID)
  set(OPTIONAL_ARGS
      ${OPTIONAL_ARGS}
      "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
      "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
      "-DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI}"
      "-DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK}"
      "-DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE}")
endif()

ExternalProject_Add(
  extern_gflags
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY "https://github.com/gflags/gflags.git"
  GIT_TAG "v2.2.2"
  PREFIX ${GFLAGS_SOURCES_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DBUILD_STATIC_LIBS=ON
             -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DBUILD_TESTING=OFF
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             -DNAMESPACE=gflags
             ${OPTIONAL_ARGS}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${GFLAGS_LIBRARIES})
if(WIN32)
  if(NOT EXISTS "${GFLAGS_INSTALL_DIR}/lib/libgflags.lib")
    add_custom_command(
      TARGET extern_gflags
      POST_BUILD
      COMMAND cmake -E copy ${GFLAGS_INSTALL_DIR}/lib/gflags_static.lib
              ${GFLAGS_INSTALL_DIR}/lib/libgflags.lib)
  endif()
endif(WIN32)
add_library(gflags STATIC IMPORTED GLOBAL)
set_property(TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARIES})
add_dependencies(gflags extern_gflags)

# On Windows (including MinGW), the Shlwapi library is used by gflags if available.
if(WIN32)
  include(CheckIncludeFileCXX)
  check_include_file_cxx("shlwapi.h" HAVE_SHLWAPI)
  if(HAVE_SHLWAPI)
    set_property(GLOBAL PROPERTY OS_DEPENDENCY_MODULES shlwapi.lib)
  endif(HAVE_SHLWAPI)
elseif(LINUX)
  set_property(GLOBAL PROPERTY OS_DEPENDENCY_MODULES pthread)
endif(WIN32)
