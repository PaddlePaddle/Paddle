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

INCLUDE(ExternalProject)

SET(CRYPTOPP_PREFIX_DIR  ${THIRD_PARTY_PATH}/cryptopp)
SET(CRYPTOPP_INSTALL_DIR ${THIRD_PARTY_PATH}/install/cryptopp)
SET(CRYPTOPP_INCLUDE_DIR "${CRYPTOPP_INSTALL_DIR}/include" CACHE PATH "cryptopp include directory." FORCE)
SET(CRYPTOPP_REPOSITORY ${GIT_URL}/weidai11/cryptopp.git)
SET(CRYPTOPP_TAG        CRYPTOPP_8_2_0)

IF(WIN32)
  SET(CRYPTOPP_LIBRARIES "${CRYPTOPP_INSTALL_DIR}/lib/cryptopp-static.lib" CACHE FILEPATH "cryptopp library." FORCE)
  # There is a compilation parameter "/FI\"winapifamily.h\"" or "/FIwinapifamily.h" can't be used correctly
  # with Ninja on Windows. The only difference between the patch file and original
  # file is that the compilation parameters are changed to '/nologo'. This
  # patch command can be removed when upgrading to a higher version.
  if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    set(CRYPTOPP_PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PADDLE_SOURCE_DIR}/patches/cryptopp/CMakeLists.txt" "<SOURCE_DIR>/")
  endif()
ELSE(WIN32)
  SET(CRYPTOPP_LIBRARIES "${CRYPTOPP_INSTALL_DIR}/lib/libcryptopp.a" CACHE FILEPATH "cryptopp library." FORCE)
ENDIF(WIN32)

IF(APPLE AND WITH_ARM)
  SET(CMAKE_CXX_FLAGS "-DCRYPTOPP_ARM_CRC32_AVAILABLE=0")
ENDIF()

set(CRYPTOPP_CMAKE_ARGS ${COMMON_CMAKE_ARGS}
                        -DBUILD_SHARED=ON
                        -DBUILD_STATIC=ON
                        -DBUILD_TESTING=OFF
                        -DCMAKE_INSTALL_LIBDIR=${CRYPTOPP_INSTALL_DIR}/lib
                        -DCMAKE_INSTALL_PREFIX=${CRYPTOPP_INSTALL_DIR}
                        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)

INCLUDE_DIRECTORIES(${CRYPTOPP_INCLUDE_DIR})

ExternalProject_Add(
    extern_cryptopp
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    GIT_REPOSITORY  ${CRYPTOPP_REPOSITORY}
    GIT_TAG         ${CRYPTOPP_TAG}
    PREFIX          ${CRYPTOPP_PREFIX_DIR}
    UPDATE_COMMAND  ""
    PATCH_COMMAND
    COMMAND ${CMAKE_COMMAND} -E remove_directory "<SOURCE_DIR>/cmake/"
    COMMAND git clone ${GIT_URL}/noloader/cryptopp-cmake "<SOURCE_DIR>/cmake"
    COMMAND cd "<SOURCE_DIR>/cmake" && git checkout tags/${CRYPTOPP_TAG} -b ${CRYPTOPP_TAG}
    COMMAND ${CMAKE_COMMAND} -E copy_directory "<SOURCE_DIR>/cmake/" "<SOURCE_DIR>/"
    COMMAND ${CRYPTOPP_PATCH_COMMAND}
    INSTALL_DIR     ${CRYPTOPP_INSTALL_DIR}
    CMAKE_ARGS ${CRYPTOPP_CMAKE_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CRYPTOPP_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    BUILD_BYPRODUCTS ${CRYPTOPP_LIBRARIES}
)

ADD_LIBRARY(cryptopp STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET cryptopp PROPERTY IMPORTED_LOCATION ${CRYPTOPP_LIBRARIES})
ADD_DEPENDENCIES(cryptopp extern_cryptopp)
