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

if (NOT WITH_ONNXRUNTIME)
  return()
endif ()

if (WITH_ARM)
  message(SEND_ERROR "The current onnxruntime backend doesn't support ARM cpu")
  return()
endif ()

INCLUDE(ExternalProject)

add_definitions(-DPADDLE_WITH_ONNXRUNTIME)

SET(ONNXRUNTIME_PROJECT        "extern_onnxruntime")
SET(ONNXRUNTIME_PREFIX_DIR     ${THIRD_PARTY_PATH}/onnxruntime)
SET(ONNXRUNTIME_SOURCE_DIR     ${THIRD_PARTY_PATH}/onnxruntime/src/${ONNXRUNTIME_PROJECT})
SET(ONNXRUNTIME_INSTALL_DIR    ${THIRD_PARTY_PATH}/install/onnxruntime)
SET(ONNXRUNTIME_INC_DIR        "${ONNXRUNTIME_INSTALL_DIR}/include" CACHE PATH "onnxruntime include directory." FORCE)
SET(ONNXRUNTIME_LIB_DIR        "${ONNXRUNTIME_INSTALL_DIR}/lib" CACHE PATH "onnxruntime lib directory." FORCE)
SET(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${ONNXRUNTIME_LIB_DIR}")


if (WIN32)
  SET(ONNXRUNTIME_URL             "https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-win-x64-1.10.0.zip")
elseif (APPLE)
  SET(ONNXRUNTIME_URL           "https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-osx-x86_64-1.10.0.tgz")
else ()
  SET(ONNXRUNTIME_URL             "https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz")
endif()


INCLUDE_DIRECTORIES(${ONNXRUNTIME_INC_DIR}) # For ONNXRUNTIME code to include internal headers.
if (WIN32)
  SET(ONNXRUNTIME_SOURCE_LIB "${ONNXRUNTIME_SOURCE_DIR}/lib/onnxruntime.dll" CACHE FILEPATH "ONNXRUNTIME source library." FORCE)
  SET(ONNXRUNTIME_SHARED_LIB "${ONNXRUNTIME_INSTALL_DIR}/lib/onnxruntime.dll" CACHE FILEPATH "ONNXRUNTIME shared library." FORCE)
  SET(ONNXRUNTIME_LIB "${ONNXRUNTIME_INSTALL_DIR}/lib/onnxruntime.lib" CACHE FILEPATH "ONNXRUNTIME static library." FORCE)
elseif (APPLE)
  SET(ONNXRUNTIME_SOURCE_LIB "${ONNXRUNTIME_SOURCE_DIR}/lib/libonnxruntime.1.10.0.dylib" CACHE FILEPATH "ONNXRUNTIME source library." FORCE)
  SET(ONNXRUNTIME_LIB "${ONNXRUNTIME_INSTALL_DIR}/lib/libonnxruntime.1.10.0.dylib" CACHE FILEPATH "ONNXRUNTIME static library." FORCE)
  SET(ONNXRUNTIME_SHARED_LIB ${ONNXRUNTIME_LIB} CACHE FILEPATH "ONNXRUNTIME shared library." FORCE)
else ()
  SET(ONNXRUNTIME_SOURCE_LIB "${ONNXRUNTIME_SOURCE_DIR}/lib/libonnxruntime.so.1.10.0" CACHE FILEPATH "ONNXRUNTIME source library." FORCE)
  SET(ONNXRUNTIME_LIB "${ONNXRUNTIME_INSTALL_DIR}/lib/libonnxruntime.so.1.10.0" CACHE FILEPATH "ONNXRUNTIME static library." FORCE)
  SET(ONNXRUNTIME_SHARED_LIB ${ONNXRUNTIME_LIB} CACHE FILEPATH "ONNXRUNTIME shared library." FORCE)
endif ()

if (WIN32)
  ExternalProject_Add(
      ${ONNXRUNTIME_PROJECT}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      URL                 ${ONNXRUNTIME_URL}
      PREFIX              ${ONNXRUNTIME_PREFIX_DIR}
      DOWNLOAD_NO_PROGRESS  1
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         ""
      UPDATE_COMMAND        ""
      INSTALL_COMMAND       ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SOURCE_LIB} ${ONNXRUNTIME_SHARED_LIB} &&
                            ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SOURCE_DIR}/lib/onnxruntime.lib ${ONNXRUNTIME_LIB} &&
                            ${CMAKE_COMMAND} -E copy_directory ${ONNXRUNTIME_SOURCE_DIR}/include ${ONNXRUNTIME_INC_DIR}
      BUILD_BYPRODUCTS      ${ONNXRUNTIME_LIB}
  )
else ()
  ExternalProject_Add(
    ${ONNXRUNTIME_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL                 ${ONNXRUNTIME_URL}
    PREFIX              ${ONNXRUNTIME_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS  1
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    UPDATE_COMMAND        ""
    INSTALL_COMMAND       ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SOURCE_LIB} ${ONNXRUNTIME_LIB} &&
                          ${CMAKE_COMMAND} -E copy_directory ${ONNXRUNTIME_SOURCE_DIR}/include ${ONNXRUNTIME_INC_DIR}
    BUILD_BYPRODUCTS      ${ONNXRUNTIME_LIB}
  )
endif()

ADD_LIBRARY(onnxruntime STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET onnxruntime PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_LIB})
ADD_DEPENDENCIES(onnxruntime ${ONNXRUNTIME_PROJECT})
