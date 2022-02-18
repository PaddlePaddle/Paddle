# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT WITH_ONNXRUNTIME)
  return()
endif()

INCLUDE(ExternalProject)

SET(PADDLE2ONNX_PROJECT        "extern_paddle2onnx")
SET(PADDLE2ONNX_PREFIX_DIR     ${THIRD_PARTY_PATH}/paddle2onnx)
SET(PADDLE2ONNX_INSTALL_DIR    ${THIRD_PARTY_PATH}/install/paddle2onnx)
SET(PADDLE2ONNX_INC_DIR        "${PADDLE2ONNX_INSTALL_DIR}/include" CACHE PATH "paddle2onnx include directory." FORCE)
SET(PADDLE2ONNX_REPOSITORY     ${GIT_URL}/PaddlePaddle/Paddle2ONNX.git)
SET(PADDLE2ONNX_TAG            cpp)

SET(LIBDIR "lib")

INCLUDE_DIRECTORIES(${PADDLE2ONNX_INC_DIR}) # For PADDLE2ONNX code to include internal headers.
IF(NOT WIN32)
    SET(PADDLE2ONNX_LIB "${PADDLE2ONNX_INSTALL_DIR}/${LIBDIR}/libpaddle2onnx.so" CACHE FILEPATH "PADDLE2ONNX library." FORCE)
ELSE()
    SET(PADDLE2ONNX_LIB "${PADDLE2ONNX_INSTALL_DIR}/${LIBDIR}/paddle2onnx.dll" CACHE FILEPATH "paddle2onnx library." FORCE)
ENDIF(NOT WIN32)

# The protoc path is required to compile onnx.
string(REPLACE "/" ";" PROTOC_BIN_PATH ${PROTOBUF_PROTOC_EXECUTABLE})
list(POP_BACK PROTOC_BIN_PATH)
list(JOIN PROTOC_BIN_PATH "/" PROTOC_BIN_PATH)


ExternalProject_Add(
    ${PADDLE2ONNX_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    GIT_REPOSITORY      ${PADDLE2ONNX_REPOSITORY}
    GIT_TAG             ${PADDLE2ONNX_TAG}
    DEPENDS             extern_protobuf
    PREFIX              ${PADDLE2ONNX_PREFIX_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DONNX_CUSTOM_PROTOC_PATH=${PROTOC_BIN_PATH}
                        -DWITH_STATIC=OFF
                        -DCMAKE_INSTALL_PREFIX=${PADDLE2ONNX_INSTALL_DIR}
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${PADDLE2ONNX_INSTALL_DIR}
    BUILD_BYPRODUCTS    ${PADDLE2ONNX_LIB}
)

ADD_LIBRARY(paddle2onnx STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET paddle2onnx PROPERTY IMPORTED_LOCATION ${PADDLE2ONNX_LIB})
ADD_DEPENDENCIES(paddle2onnx ${PADDLE2ONNX_PROJECT})
