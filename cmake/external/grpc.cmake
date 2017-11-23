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
#

IF(MOBILE_INFERENCE)
    return()
ENDIF()

include (ExternalProject)

SET(GRPC_SOURCES_DIR ${THIRD_PARTY_PATH}/grpc)
SET(GRPC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/grpc)
SET(GRPC_INCLUDE_DIR "${GRPC_INSTALL_DIR}/include/" CACHE PATH "grpc include directory." FORCE)
SET(GRPC_CPP_PLUGIN "${GRPC_INSTALL_DIR}/bin/grpc_cpp_plugin" CACHE FILEPATH "GRPC_CPP_PLUGIN" FORCE)

ExternalProject_Add(
    extern_grpc
    DEPENDS protobuf zlib
    GIT_REPOSITORY "https://github.com/grpc/grpc.git"
    GIT_TAG "v1.7.x"
    PREFIX          ${GRPC_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND   make
    INSTALL_COMMAND make prefix=${GRPC_INSTALL_DIR} install
)

# FIXME(typhoonzero): hack to get static lib path, try a better way like merge them.
ADD_LIBRARY(grpc++_unsecure STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET grpc++_unsecure PROPERTY IMPORTED_LOCATION
             "${GRPC_INSTALL_DIR}/lib/libgrpc++_unsecure.a")

ADD_LIBRARY(grpc++ STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET grpc++ PROPERTY IMPORTED_LOCATION
            "${GRPC_INSTALL_DIR}/lib/libgrpc++.a")
ADD_LIBRARY(gpr STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gpr PROPERTY IMPORTED_LOCATION
            "${GRPC_INSTALL_DIR}/lib/libgpr.a")

ADD_LIBRARY(grpc_unsecure STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET grpc_unsecure PROPERTY IMPORTED_LOCATION
            "${GRPC_INSTALL_DIR}/lib/libgrpc_unsecure.a")

include_directories(${GRPC_INCLUDE_DIR})
ADD_DEPENDENCIES(grpc++_unsecure extern_grpc)

