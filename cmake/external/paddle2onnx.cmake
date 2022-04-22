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

if (WITH_ARM)
  message(SEND_ERROR "The current onnxruntime backend doesn't support ARM cpu")
  return()
endif ()

INCLUDE(ExternalProject)

SET(PADDLE2ONNX_PROJECT        "extern_paddle2onnx")
SET(PADDLE2ONNX_PREFIX_DIR     ${THIRD_PARTY_PATH}/paddle2onnx)
SET(PADDLE2ONNX_INSTALL_DIR    ${THIRD_PARTY_PATH}/install/paddle2onnx)
SET(PADDLE2ONNX_INC_DIR        "${PADDLE2ONNX_INSTALL_DIR}/include" CACHE PATH "paddle2onnx include directory." FORCE)
SET(PADDLE2ONNX_REPOSITORY     ${GIT_URL}/PaddlePaddle/Paddle2ONNX.git)
SET(PADDLE2ONNX_TAG            cpp)
SET(LIBDIR "lib")
SET(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${PADDLE2ONNX_INSTALL_DIR}/${LIBDIR}")

INCLUDE_DIRECTORIES(${PADDLE2ONNX_INC_DIR}) # For PADDLE2ONNX code to include internal headers.
if(WIN32)
    SET(PADDLE2ONNX_LIB "${PADDLE2ONNX_INSTALL_DIR}/${LIBDIR}/paddle2onnx.lib" CACHE FILEPATH "paddle2onnx static library." FORCE)
    SET(PADDLE2ONNX_SHARED_LIB "${PADDLE2ONNX_INSTALL_DIR}/${LIBDIR}/paddle2onnx.dll" CACHE FILEPATH "paddle2onnx shared library." FORCE)
elseif(APPLE)
    SET(PADDLE2ONNX_LIB "${PADDLE2ONNX_INSTALL_DIR}/${LIBDIR}/libpaddle2onnx.dylib" CACHE FILEPATH "PADDLE2ONNX library." FORCE)
else()
    SET(PADDLE2ONNX_LIB "${PADDLE2ONNX_INSTALL_DIR}/${LIBDIR}/libpaddle2onnx.so" CACHE FILEPATH "PADDLE2ONNX library." FORCE)
endif(WIN32)


# The protoc path is required to compile onnx.
string(REPLACE "/" ";" PROTOC_BIN_PATH ${PROTOBUF_PROTOC_EXECUTABLE})
list(POP_BACK PROTOC_BIN_PATH)
list(JOIN PROTOC_BIN_PATH "/" PROTOC_BIN_PATH)


set(PADDLE2ONNX_OPTIONAL_ARGS
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -std=c++11"
      -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
      -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
      -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
      -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
      -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
      -DONNX_CUSTOM_PROTOC_PATH=${PROTOC_BIN_PATH}
      -DWITH_STATIC=OFF
      -DCMAKE_INSTALL_PREFIX=${PADDLE2ONNX_INSTALL_DIR}
      -DCMAKE_INSTALL_LIBDIR=${PADDLE2ONNX_INSTALL_DIR}/${LIBDIR}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
      ${EXTERNAL_OPTIONAL_ARGS}
)

if (WITH_PYTHON)
  set(PADDLE2ONNX_OPTIONAL_ARGS ${PADDLE2ONNX_OPTIONAL_ARGS}
    -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
    -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
    -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
  )
endif ()


ExternalProject_Add(
    ${PADDLE2ONNX_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    GIT_REPOSITORY      ${PADDLE2ONNX_REPOSITORY}
    GIT_TAG             ${PADDLE2ONNX_TAG}
    DEPENDS             protobuf
    PREFIX              ${PADDLE2ONNX_PREFIX_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS       ${PADDLE2ONNX_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${PADDLE2ONNX_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    BUILD_BYPRODUCTS    ${PADDLE2ONNX_LIB}
)

ADD_LIBRARY(paddle2onnx STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET paddle2onnx PROPERTY IMPORTED_LOCATION ${PADDLE2ONNX_LIB})
ADD_DEPENDENCIES(paddle2onnx ${PADDLE2ONNX_PROJECT})
