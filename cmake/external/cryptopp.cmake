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

set(CRYPTOPP_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/cryptopp)
set(CRYPTOPP_CMAKE_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/cryptopp-cmake)
set(CRYPTOPP_PREFIX_DIR ${THIRD_PARTY_PATH}/cryptopp)
set(CRYPTOPP_INSTALL_DIR ${THIRD_PARTY_PATH}/install/cryptopp)
set(CRYPTOPP_INCLUDE_DIR
    "${CRYPTOPP_INSTALL_DIR}/include"
    CACHE PATH "cryptopp include directory." FORCE)
set(CRYPTOPP_TAG CRYPTOPP_8_2_0)

if(WIN32)
  set(CRYPTOPP_LIBRARIES
      "${CRYPTOPP_INSTALL_DIR}/lib/cryptopp-static.lib"
      CACHE FILEPATH "cryptopp library." FORCE)
  # There is a compilation parameter "/FI\"winapifamily.h\"" or "/FIwinapifamily.h" can't be used correctly
  # with Ninja on Windows. The only difference between the patch file and original
  # file is that the compilation parameters are changed to '/nologo'. This
  # patch command can be removed when upgrading to a higher version.
  if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    set(CRYPTOPP_PATCH_COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${PADDLE_SOURCE_DIR}/patches/cryptopp/CMakeLists.txt" "<SOURCE_DIR>/")
  endif()
else()
  set(CRYPTOPP_LIBRARIES
      "${CRYPTOPP_INSTALL_DIR}/lib/libcryptopp.a"
      CACHE FILEPATH "cryptopp library." FORCE)
endif()

if(APPLE AND WITH_ARM)
  set(CMAKE_CXX_FLAGS "-DCRYPTOPP_ARM_CRC32_AVAILABLE=0")
endif()

set(CRYPTOPP_CMAKE_ARGS
    ${COMMON_CMAKE_ARGS}
    -DBUILD_SHARED=ON
    -DBUILD_STATIC=ON
    -DBUILD_TESTING=OFF
    -DCMAKE_INSTALL_LIBDIR=${CRYPTOPP_INSTALL_DIR}/lib
    -DCMAKE_INSTALL_PREFIX=${CRYPTOPP_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})

include_directories(${CRYPTOPP_INCLUDE_DIR})

ExternalProject_Add(
  extern_cryptopp
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  PREFIX ${CRYPTOPP_PREFIX_DIR}
  SOURCE_DIR ${CRYPTOPP_SOURCE_DIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND
  COMMAND ${CMAKE_COMMAND} -E copy "${CRYPTOPP_CMAKE_SOURCE_DIR}/CMakeLists.txt"
          "<SOURCE_DIR>/CMakeLists.txt"
  COMMAND
    ${CMAKE_COMMAND} -E copy
    "${CRYPTOPP_CMAKE_SOURCE_DIR}/cryptopp-config.cmake"
    "<SOURCE_DIR>/cryptopp-config.cmake"
  COMMAND ${CRYPTOPP_PATCH_COMMAND}
  INSTALL_DIR ${CRYPTOPP_INSTALL_DIR}
  CMAKE_ARGS ${CRYPTOPP_CMAKE_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${CRYPTOPP_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${CRYPTOPP_LIBRARIES})

add_library(cryptopp STATIC IMPORTED GLOBAL)
set_property(TARGET cryptopp PROPERTY IMPORTED_LOCATION ${CRYPTOPP_LIBRARIES})
add_dependencies(cryptopp extern_cryptopp)
