# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

set(XXHASH_PREFIX_DIR ${THIRD_PARTY_PATH}/xxhash)
set(XXHASH_SOURCE_DIR ${THIRD_PARTY_PATH}/xxhash/src/extern_xxhash)
set(XXHASH_INSTALL_DIR ${THIRD_PARTY_PATH}/install/xxhash)
set(XXHASH_INCLUDE_DIR "${XXHASH_INSTALL_DIR}/include")
set(XXHASH_REPOSITORY  ${GIT_URL}/Cyan4973/xxHash.git)
set(XXHASH_TAG         v0.6.5)

INCLUDE_DIRECTORIES(${XXHASH_INCLUDE_DIR})

IF(APPLE)
  SET(BUILD_CMD sed -i \"\" "s/-Wstrict-prototypes -Wundef/-Wstrict-prototypes -Wundef -fPIC/g" ${XXHASH_SOURCE_DIR}/Makefile && make lib)
ELSEIF(UNIX)
  SET(BUILD_CMD sed -i "s/-Wstrict-prototypes -Wundef/-Wstrict-prototypes -Wundef -fPIC/g" ${XXHASH_SOURCE_DIR}/Makefile && make lib)
ENDIF()

if (WIN32)
  set(XXHASH_LIBRARIES "${XXHASH_INSTALL_DIR}/lib/xxhash.lib")
  set(XXHASH_CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4710 /wd4711")
  set(XXHASH_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4710 /wd4711")
else()
  set(XXHASH_LIBRARIES "${XXHASH_INSTALL_DIR}/lib/libxxhash.a")
  set(XXHASH_CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
  set(XXHASH_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif ()

if(WIN32)
  ExternalProject_Add(
      extern_xxhash
      ${EXTERNAL_PROJECT_LOG_ARGS}
      ${SHALLOW_CLONE}
      GIT_REPOSITORY   ${XXHASH_REPOSITORY}
      GIT_TAG          ${XXHASH_TAG}
      PREFIX           ${XXHASH_PREFIX_DIR}
      UPDATE_COMMAND   ""
      PATCH_COMMAND    ""
      CONFIGURE_COMMAND
                      ${CMAKE_COMMAND} ${XXHASH_SOURCE_DIR}/cmake_unofficial
                      -DCMAKE_INSTALL_PREFIX:PATH=${XXHASH_INSTALL_DIR}
                      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
                      -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
                      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                      -DBUILD_XXHSUM=OFF
                      -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                      -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
                      -DBUILD_SHARED_LIBS=OFF
                      -DCMAKE_CXX_FLAGS=${XXHASH_CMAKE_CXX_FLAGS}
                      -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                      -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                      -DCMAKE_C_FLAGS=${XXHASH_CMAKE_C_FLAGS}
                      -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                      -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                      ${OPTIONAL_CACHE_ARGS}
      TEST_COMMAND      ""
      BUILD_BYPRODUCTS ${XXHASH_LIBRARIES}
  )
else()
  ExternalProject_Add(
      extern_xxhash
      ${EXTERNAL_PROJECT_LOG_ARGS}
      GIT_REPOSITORY   ${XXHASH_REPOSITORY}
      GIT_TAG          ${XXHASH_TAG}
      PREFIX           ${XXHASH_PREFIX_DIR}
      UPDATE_COMMAND    ""
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE   1
      BUILD_COMMAND     ${BUILD_CMD}
      INSTALL_COMMAND   make PREFIX=${XXHASH_INSTALL_DIR} install
      TEST_COMMAND      ""
      BUILD_BYPRODUCTS  ${XXHASH_LIBRARIES}
  )
endif()

add_library(xxhash STATIC IMPORTED GLOBAL)
set_property(TARGET xxhash PROPERTY IMPORTED_LOCATION ${XXHASH_LIBRARIES})
include_directories(${XXHASH_INCLUDE_DIR})
add_dependencies(xxhash extern_xxhash)
