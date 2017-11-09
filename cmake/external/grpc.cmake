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

include (ExternalProject)

SET(GRPC_SOURCES_DIR ${THIRD_PARTY_PATH}/grpc)
SET(GRPC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/grpc)
SET(GRPC_INCLUDE_DIR "${GRPC_INSTALL_DIR}/include" CACHE PATH "grpc include directory." FORCE)

ExternalProject_Add(grpc
    DEPENDS protobuf zlib
    GIT_REPOSITORY "https://github.com/grpc/grpc.git"
    GIT_TAG "v1.7.x"
    PREFIX          ${GRPC_SOURCES_DIR}
    UPDATE_COMMAND  ""
    # TODO(jhseu): Remove this PATCH_COMMAND once grpc removes the dependency
    # on "grpc" from the "grpc++_unsecure" rule.
    #PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/patches/grpc/CMakeLists.txt ${GRPC_SOURCES_DIR}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target grpc++_unsecure
    COMMAND ${CMAKE_COMMAND} --build . --config Release --target grpc_cpp_plugin
    INSTALL_COMMAND ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_INSTALL_PREFIX=${GRPC_INSTALL_DIR}
                    -DCMAKE_INSTALL_LIBDIR=${GRPC_INSTALL_DIR}/lib
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DWITH_GFLAGS=ON
                    -Dgflags_DIR=${GRPC_INSTALL_DIR}/lib/cmake/gflags
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DPROTOBUF_INCLUDE_DIRS:STRING=${PROTOBUF_INCLUDE_DIRS}
        -DPROTOBUF_LIBRARIES:STRING=${protobuf_STATIC_LIBRARIES}
        -DZLIB_ROOT:STRING=${ZLIB_INSTALL}
	      -DgRPC_SSL_PROVIDER:STRING=NONE
)

# grpc/src/core/ext/census/tracing.c depends on the existence of openssl/rand.h.
ExternalProject_Add_Step(grpc copy_rand
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/patches/grpc/rand.h ${GRPC_INCLUDE_DIR}/openssl/rand.h
    DEPENDEES patch
    DEPENDERS build
)