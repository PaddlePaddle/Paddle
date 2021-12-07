# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

SET(UTF8PROC_PREFIX_DIR    ${THIRD_PARTY_PATH}/utf8proc)
SET(UTF8PROC_INSTALL_DIR   ${THIRD_PARTY_PATH}/install/utf8proc)
# As we add extra features for utf8proc, we use the non-official repo
SET(UTF8PROC_REPOSITORY    ${GIT_URL}/JuliaStrings/utf8proc.git)
SET(UTF8PROC_TAG           v2.6.1)

IF(WIN32)
  SET(UTF8PROC_LIBRARIES     "${UTF8PROC_INSTALL_DIR}/lib/utf8proc_static.lib")
  add_definitions(-DUTF8PROC_STATIC)
ELSE(WIN32)
  SET(UTF8PROC_LIBRARIES     "${UTF8PROC_INSTALL_DIR}/lib/libutf8proc.a")
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${UTF8PROC_INSTALL_DIR}/include)

ExternalProject_Add(
  extern_utf8proc
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  GIT_REPOSITORY        ${UTF8PROC_REPOSITORY}
  GIT_TAG               ${UTF8PROC_TAG}
  PREFIX                ${UTF8PROC_PREFIX_DIR}
  UPDATE_COMMAND        ""
  CMAKE_ARGS            -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                        -DBUILD_SHARED=ON
                        -DBUILD_STATIC=ON
                        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                        -DCMAKE_INSTALL_PREFIX:PATH=${UTF8PROC_INSTALL_DIR}
                        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  BUILD_BYPRODUCTS     ${UTF8PROC_LIBRARIES}
)

ADD_LIBRARY(utf8proc STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET utf8proc PROPERTY IMPORTED_LOCATION ${UTF8PROC_LIBRARIES})
ADD_DEPENDENCIES(utf8proc extern_utf8proc)
