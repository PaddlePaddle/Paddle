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

SET(GFLAGS_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gflags)
SET(GFLAGS_INSTALL_DIR ${PROJECT_BINARY_DIR}/gflags)

ExternalProject_Add(
    gflags
    GIT_REPOSITORY  "https://github.com/gflags/gflags.git"
    PREFIX          ${GFLAGS_SOURCES_DIR}
    CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
    CMAKE_ARGS      -DBUILD_TESTING=OFF
    LOG_DOWNLOAD    =ON
    UPDATE_COMMAND  ""
)

SET(GFLAGS_INCLUDE_DIR "${GFLAGS_INSTALL_DIR}/include" CACHE PATH "gflags include directory." FORCE)
INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE_DIR})

IF(WIN32)
    set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/gflags.lib" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
ELSE(WIN32)
    set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/libgflags.a" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
ENDIF(WIN32)

LIST(APPEND external_project_dependencies gflags)
