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

SET(UTF8PROC_PROJECT       "extern_utf8proc")
SET(UTF8PROC_PREFIX_DIR    ${THIRD_PARTY_PATH}/utf8proc)
SET(UTF8PROC_SOURCE_DIR    ${THIRD_PARTY_PATH}/utf8proc/src/extern_utf8proc/utf8proc)
SET(UTF8PROC_INSTALL_DIR   ${THIRD_PARTY_PATH}/install/utf8proc)
SET(UTF8PROC_INCLUDE_DIR   "${UTF8PROC_INSTALL_DIR}/include" CACHE PATH "utf8proc include directory." FORCE)
SET(UTF8PROC_LIBRARY_DIR   "${UTF8PROC_INSTALL_DIR}/lib" CACHE PATH "utf8proc library directory." FORCE)
# As we add extra features for utf8proc, we use the non-official repo
SET(UTF8PROC_REPOSITORY    ${GIT_URL}/JuliaStrings/utf8proc.git)
SET(UTF8PROC_TAG           v2.6.1)
SET(UTF8PROC_LIBRARIES     "${UTF8PROC_INSTALL_DIR}/lib/libutf8proc.a" CACHE FILEPATH "utf8proc library." FORCE)

INCLUDE_DIRECTORIES(${UTF8PROC_INCLUDE_DIR})

cache_third_party(extern_utf8proc
    REPOSITORY    ${UTF8PROC_REPOSITORY}
    TAG           ${UTF8PROC_TAG}
    DIR           ${UTF8PROC_SOURCE_DIR})

if(WITH_ASCEND OR WITH_ASCEND_CL)
  ExternalProject_Add(
      extern_utf8proc
      ${EXTERNAL_PROJECT_LOG_ARGS}
      ${SHALLOW_CLONE}
      "${UTFPROC_DOWNLOAD_CMD}"
      PREFIX                "${UTF8PROC_PREFIX_DIR}"
      SOURCE_DIR            "${UTF8PROC_SOURCE_DIR}"
      UPDATE_COMMAND        ""
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         mkdir -p ${UTF8PROC_SOURCE_DIR}/build
          && cd ${UTF8PROC_SOURCE_DIR}/build && cmake .. -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} && make
          && mkdir -p ${UTF8PROC_LIBRARY_DIR} ${UTF8PROC_INCLUDE_DIR}
      INSTALL_COMMAND      ${CMAKE_COMMAND} -E copy ${UTF8PROC_SOURCE_DIR}/build/libutf8proc.a ${UTF8PROC_LIBRARY_DIR}
      COMMAND              ${CMAKE_COMMAND} -E copy ${UTF8PROC_SOURCE_DIR}/utf8proc.h ${UTF8PROC_INCLUDE_DIR}
      BUILD_BYPRODUCTS     ${UTF8PROC_LIBRARIES}
  )
else()
  ExternalProject_Add(
      extern_utf8proc
      ${EXTERNAL_PROJECT_LOG_ARGS}
      ${SHALLOW_CLONE}
      "${UTFPROC_DOWNLOAD_CMD}"
      PREFIX                "${UTF8PROC_PREFIX_DIR}"
      SOURCE_DIR            "${UTF8PROC_SOURCE_DIR}"
      UPDATE_COMMAND        ""
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         mkdir -p ${UTF8PROC_SOURCE_DIR}/build
          && cd ${UTF8PROC_SOURCE_DIR}/build && cmake .. && make
          && mkdir -p ${UTF8PROC_LIBRARY_DIR} ${UTF8PROC_INCLUDE_DIR}
      INSTALL_COMMAND      ${CMAKE_COMMAND} -E copy ${UTF8PROC_SOURCE_DIR}/build/libutf8proc.a ${UTF8PROC_LIBRARY_DIR}
      COMMAND              ${CMAKE_COMMAND} -E copy ${UTF8PROC_SOURCE_DIR}/utf8proc.h ${UTF8PROC_INCLUDE_DIR}
      BUILD_BYPRODUCTS     ${UTF8PROC_LIBRARIES}
  )
endif()


ADD_LIBRARY(utf8proc STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET utf8proc PROPERTY IMPORTED_LOCATION ${UTF8PROC_LIBRARIES})
ADD_DEPENDENCIES(utf8proc ${UTF8PROC_PROJECT})
