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

INCLUDE(ExternalProject)

SET(GLOG_SOURCES_DIR ${THIRD_PARTY_PATH}/glog)
SET(GLOG_INSTALL_DIR ${THIRD_PARTY_PATH}/install/glog)
SET(GLOG_INCLUDE_DIR "${GLOG_INSTALL_DIR}/include" CACHE PATH "glog include directory." FORCE)

IF(WIN32)
  SET(GLOG_LIBRARIES "${GLOG_INSTALL_DIR}/lib/libglog.lib" CACHE FILEPATH "glog library." FORCE)
  SET(GLOG_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4530")
ELSE(WIN32)
  SET(GLOG_LIBRARIES "${GLOG_INSTALL_DIR}/lib/libglog.a" CACHE FILEPATH "glog library." FORCE)
  SET(GLOG_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${GLOG_INCLUDE_DIR})

SET(GLOG_REPOSITORY "https://github.com/google/glog.git")
SET(GLOG_TAG "v0.3.5")

ExternalProject_Add(
    extern_glog
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS gflags
    GIT_REPOSITORY  ${GLOG_REPOSITORY}
    GIT_TAG         ${GLOG_TAG}
    PREFIX          ${GLOG_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${GLOG_CMAKE_CXX_FLAGS}
                    -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                    -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                    -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                    -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR}
                    -DCMAKE_INSTALL_LIBDIR=${GLOG_INSTALL_DIR}/lib
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DWITH_GFLAGS=ON
                    -Dgflags_DIR=${GFLAGS_INSTALL_DIR}/lib/cmake/gflags
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${GLOG_INSTALL_DIR}
                     -DCMAKE_INSTALL_LIBDIR:PATH=${GLOG_INSTALL_DIR}/lib
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)
IF(WIN32)
  IF(NOT EXISTS "${GLOG_INSTALL_DIR}/lib/libglog.lib")
    add_custom_command(TARGET extern_glog POST_BUILD
    COMMAND cmake -E copy ${GLOG_INSTALL_DIR}/lib/glog.lib ${GLOG_INSTALL_DIR}/lib/libglog.lib
  )
  ENDIF()
ENDIF(WIN32)

ADD_LIBRARY(glog STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET glog PROPERTY IMPORTED_LOCATION ${GLOG_LIBRARIES})
ADD_DEPENDENCIES(glog extern_glog gflags)
LINK_LIBRARIES(glog gflags)

LIST(APPEND external_project_dependencies glog)
