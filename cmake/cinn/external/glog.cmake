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

set(GLOG_SOURCES_DIR ${CINN_THIRD_PARTY_PATH}/glog)
set(GLOG_INSTALL_DIR ${CINN_THIRD_PARTY_PATH}/install/glog)
set(GLOG_INCLUDE_DIR
    "${GLOG_INSTALL_DIR}/include"
    CACHE PATH "glog include directory." FORCE)

if(WIN32)
  set(GLOG_LIBRARIES
      "${GLOG_INSTALL_DIR}/lib/libglog.lib"
      CACHE FILEPATH "glog library." FORCE)
  set(GLOG_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4530")
else(WIN32)
  set(GLOG_LIBRARIES
      "${GLOG_INSTALL_DIR}/lib/libglog.a"
      CACHE FILEPATH "glog library." FORCE)
  set(GLOG_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif(WIN32)

include_directories(${GLOG_INCLUDE_DIR})

set(GLOG_REPOSITORY "https://github.com/google/glog.git")
set(GLOG_TAG "v0.4.0")

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
  extern_glog
  ${EXTERNAL_PROJECT_LOG_ARGS}
  DEPENDS gflags
  GIT_REPOSITORY ${GLOG_REPOSITORY}
  GIT_TAG ${GLOG_TAG}
  PREFIX ${GLOG_SOURCES_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS ${OPTIONAL_ARGS}
             -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR}
             -DCMAKE_INSTALL_LIBDIR=${GLOG_INSTALL_DIR}/lib
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DWITH_GFLAGS=OFF
             -DBUILD_TESTING=OFF
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${GLOG_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR:PATH=${GLOG_INSTALL_DIR}/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${GLOG_LIBRARIES})
if(WIN32)
  if(NOT EXISTS "${GLOG_INSTALL_DIR}/lib/libglog.lib")
    add_custom_command(
      TARGET extern_glog
      POST_BUILD
      COMMAND cmake -E copy ${GLOG_INSTALL_DIR}/lib/glog.lib
              ${GLOG_INSTALL_DIR}/lib/libglog.lib)
  endif()
endif(WIN32)

add_library(glog STATIC IMPORTED GLOBAL)
set_property(TARGET glog PROPERTY IMPORTED_LOCATION ${GLOG_LIBRARIES})
add_dependencies(glog extern_glog gflags)
link_libraries(glog gflags)
