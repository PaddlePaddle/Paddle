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
#

IF(MOBILE_INFERENCE OR NOT WITH_DISTRIBUTE)
    return()
ENDIF()

include (ExternalProject)

SET(OPENSSL_SOURCES_DIR ${THIRD_PARTY_PATH}/openssl)
SET(OPENSSL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openssl)
SET(OPENSSL_INCLUDE_DIR "${OPENSSL_INSTALL_DIR}/include/" CACHE PATH "openssl include directory." FORCE)
SET(OPENSSL_LIBRARIES "${OPENSSL_INSTALL_DIR}/lib/libssl.a" CACHE FILEPATH "ssl library." FORCE)
SET(CRYPTO_LIBRARIES "${OPENSSL_INSTALL_DIR}/lib/libcrypto.a" CACHE FILEPATH "ssl library." FORCE)

ExternalProject_Add(
    extern_openssl
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/openssl/openssl"
    GIT_TAG         "c35608e5422d2718868d88439e22369d4aabb7c6"
    PREFIX          ${OPENSSL_SOURCES_DIR}
    UPDATE_COMMAND  ""
    INSTALL_DIR ${OPENSSL_INSTALL_DIR}
    CONFIGURE_COMMAND ./config -fPIC no-shared --prefix=${OPENSSL_INSTALL_DIR}
    BUILD_IN_SOURCE 1
    BUILD_COMMAND $(MAKE) -j 12
    INSTALL_COMMAND $(MAKE) install
)

ADD_DEPENDENCIES(extern_openssl zlib)
ADD_LIBRARY(ssl STATIC IMPORTED GLOBAL)
ADD_LIBRARY(crypto STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET ssl PROPERTY IMPORTED_LOCATION ${OPENSSL_LIBRARIES})
SET_PROPERTY(TARGET crypto PROPERTY IMPORTED_LOCATION ${CRYPTO_LIBRARIES})
ADD_DEPENDENCIES(ssl extern_ssl)


LIST(APPEND external_project_dependencies ssl)
