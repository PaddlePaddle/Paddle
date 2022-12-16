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

include(ExternalProject)

set(ZLIB_PREFIX_DIR ${THIRD_PARTY_PATH}/zlib)
set(ZLIB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/zlib)
set(ZLIB_ROOT
    ${ZLIB_INSTALL_DIR}
    CACHE FILEPATH "zlib root directory." FORCE)
set(ZLIB_INCLUDE_DIR
    "${ZLIB_INSTALL_DIR}/include"
    CACHE PATH "zlib include directory." FORCE)
set(ZLIB_REPOSITORY ${GIT_URL}/madler/zlib.git)
set(ZLIB_TAG v1.2.8)

include_directories(${ZLIB_INCLUDE_DIR}
)# For zlib code to include its own headers.
include_directories(${THIRD_PARTY_PATH}/install
)# For Paddle code to include zlib.h.

if(WIN32)
  set(ZLIB_LIBRARIES
      "${ZLIB_INSTALL_DIR}/lib/zlibstatic.lib"
      CACHE FILEPATH "zlib library." FORCE)
else()
  set(ZLIB_LIBRARIES
      "${ZLIB_INSTALL_DIR}/lib/libz.a"
      CACHE FILEPATH "zlib library." FORCE)
endif()

ExternalProject_Add(
  extern_zlib
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${ZLIB_REPOSITORY}
  GIT_TAG ${ZLIB_TAG}
  PREFIX ${ZLIB_PREFIX_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR}
             -DBUILD_SHARED_LIBS=OFF
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_MACOSX_RPATH=ON
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${ZLIB_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${ZLIB_LIBRARIES})

add_library(zlib STATIC IMPORTED GLOBAL)
set_property(TARGET zlib PROPERTY IMPORTED_LOCATION ${ZLIB_LIBRARIES})
add_dependencies(zlib extern_zlib)
