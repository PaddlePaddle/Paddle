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

INCLUDE(ExternalProject)

SET(GFLAGS_SOURCES_DIR ${THIRD_PARTY_PATH}/gflags)
SET(GFLAGS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gflags)
SET(GFLAGS_INCLUDE_DIR "${GFLAGS_INSTALL_DIR}/include" CACHE PATH "gflags include directory." FORCE)
IF(WIN32)
  set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/gflags.lib" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
ELSE(WIN32)
  set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/libgflags.a" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE_DIR})

ExternalProject_Add(
    extern_gflags
    ${EXTERNAL_PROJECT_LOG_ARGS}
    # TODO(yiwang): The annoying warnings mentioned in
    # https://github.com/PaddlePaddle/Paddle/issues/3277 are caused by
    # gflags.  I fired a PR https://github.com/gflags/gflags/pull/230
    # to fix it.  Before it gets accepted by the gflags team, we use
    # my personal fork, which contains above fix, temporarily.  Let's
    # change this back to the official Github repo once my PR is
    # merged.
    GIT_REPOSITORY  "https://github.com/wangkuiyi/gflags.git"
    PREFIX          ${GFLAGS_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=Release
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=Release
)

ADD_LIBRARY(gflags STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARIES})
ADD_DEPENDENCIES(gflags extern_gflags)

LIST(APPEND external_project_dependencies gflags)

IF(WITH_C_API)
  INSTALL(DIRECTORY ${GFLAGS_INCLUDE_DIR} DESTINATION third_party/gflags)
  IF(ANDROID)
    INSTALL(FILES ${GFLAGS_LIBRARIES} DESTINATION third_party/gflags/lib/${ANDROID_ABI})
  ELSE()
    INSTALL(FILES ${GFLAGS_LIBRARIES} DESTINATION third_party/gflags/lib)
  ENDIF()
ENDIF()
