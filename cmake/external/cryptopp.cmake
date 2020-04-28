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
SET(CRYPTOPP_REPOSITORY https://github.com/weidai11/cryptopp.git)
SET(CRYPTOPP_TAG        CRYPTOPP_8_2_0)
SET(BUILD_COMMAND make)
SET(INSTALL_COMMAND make PREFIX=${CRYPTOPP_INSTALL_DIR} install)

IF(WIN32)
  SET(CRYPTOPP_LIBRARIES "${CRYPTOPP_INSTALL_DIR}/lib/cryptopp.lib" CACHE FILEPATH "cryptopp library." FORCE)
  SET(CRYPTOPP_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4530")
ELSE(WIN32)
  SET(CRYPTOPP_LIBRARIES "${CRYPTOPP_INSTALL_DIR}/lib/libcryptopp.a" CACHE FILEPATH "cryptopp library." FORCE)
  SET(CRYPTOPP_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${CRYPTOPP_INCLUDE_DIR})

cache_third_party(extern_cryptopp
    REPOSITORY   ${CRYPTOPP_REPOSITORY}
    TAG          ${CRYPTOPP_TAG})

ExternalProject_Add(
    extern_cryptopp
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${CRYPTOPP_DOWNLOAD_CMD}"
    PREFIX          ${CRYPTOPP_PREFIX_DIR}
    SOURCE_DIR      ${CRYPTOPP_SOURCE_DIR}
    BUILD_COMMAND   ${BUILD_COMMAND}
    UPDATE_COMMAND    ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE   1
    INSTALL_COMMAND   ${INSTALL_COMMAND}
    TEST_COMMAND      ""
)

ADD_LIBRARY(cryptopp STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET cryptopp PROPERTY IMPORTED_LOCATION ${CRYPTOPP_LIBRARIES})
ADD_DEPENDENCIES(cryptopp extern_cryptopp)
