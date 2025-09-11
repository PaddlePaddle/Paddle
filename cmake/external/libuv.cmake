# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

set(LIBUV_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/libuv)
set(LIBUV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/libuv)

if(WIN32)
  set(LIBUV_LIBRARIES ${LIBUV_INSTALL_DIR}/lib/libuv.lib)
  set(LIBUV_INCLUDE_DIR ${LIBUV_INSTALL_DIR}/include)

  if(MSVC_STATIC_CRT)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
      set(LIDUV_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")
    else()
      set(LIDUV_MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()

    set(LIBUV_CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(LIBUV_CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    foreach(
      flag_var
      CMAKE_CXX_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
      CMAKE_C_FLAGS
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
  else()
    if(CMAKE_BUILD_TYPE MATCHES Debug)
      set(LIDUV_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
    else()
      set(LIDUV_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()

    set(LIBUV_CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(LIBUV_CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    foreach(
      flag_var
      CMAKE_CXX_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
      CMAKE_C_FLAGS
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
  endif()
else()
  # Unix-like platform (Linux or macOS)
  set(LIBUV_LIBRARIES ${LIBUV_INSTALL_DIR}/lib/libuv.a)
  set(LIBUV_INCLUDE_DIR ${LIBUV_INSTALL_DIR}/include)
endif()

ExternalProject_Add(
  extern_libuv
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${LIBUV_SOURCE_DIR}
  BINARY_DIR ${LIBUV_SOURCE_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${LIBUV_INSTALL_DIR}
             -DCMAKE_INSTALL_LIBDIR=${LIBUV_INSTALL_DIR}/lib
             -DCMAKE_MSVC_RUNTIME_LIBRARY=${LIDUV_MSVC_RUNTIME_LIBRARY}
             -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_C_FLAGS_RELEASE=${LIBUV_CMAKE_CXX_FLAGS_RELEASE}
             -DCMAKE_C_FLAGS_DEBUG={LIBUV_CMAKE_CXX_FLAGS_DEBUG}
             -DBUILD_SHARED_LIBS=OFF
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DBUILD_TESTING=OFF
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${LIBUV_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  # output
  BUILD_BYPRODUCTS ${LIBUV_LIBRARIES})

add_library(libuv STATIC IMPORTED)
add_dependencies(libuv extern_libuv)

set_target_properties(libuv PROPERTIES IMPORTED_LOCATION ${LIBUV_LIBRARIES})
if(WIN32)
  set_target_properties(
    libuv PROPERTIES INTERFACE_LINK_LIBRARIES
                     "ws2_32;psapi;iphlpapi;userenv;advapi32")
endif()

include_directories(${LIBUV_INCLUDE_DIR})
