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

SET(GTEST_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gtest)
SET(GTEST_INSTALL_DIR ${PROJECT_BINARY_DIR}/gtest)

ExternalProject_Add(
    gtest
    GIT_REPOSITORY  "https://github.com/google/googletest.git"
    GIT_TAG         "release-1.8.0"
    PREFIX          ${GTEST_SOURCES_DIR}
    CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${GTEST_INSTALL_DIR}
    CMAKE_ARGS      -DBUILD_GMOCK=ON
    CMAKE_ARGS      -Dgtest_disable_pthreads=ON
    CMAKE_ARGS      -Dgtest_force_shared_crt=ON
    LOG_DOWNLOAD    =ON
    UPDATE_COMMAND  ""
)

SET(GTEST_INCLUDE_DIR "${GTEST_INSTALL_DIR}/include" CACHE PATH "gtest include directory." FORCE)
INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})

IF(WIN32)
    set(GTEST_LIBRARIES
        "${GTEST_INSTALL_DIR}/lib/gtest.lib"
        "${GTEST_INSTALL_DIR}/lib/gtest_main.lib" CACHE FILEPATH "gtest libraries." FORCE)
ELSE(WIN32)
    set(GTEST_LIBRARIES
        "${GTEST_INSTALL_DIR}/lib/libgtest.a"
        "${GTEST_INSTALL_DIR}/lib/libgtest_main.a" CACHE FILEPATH "gtest libraries." FORCE)
ENDIF(WIN32)

ENABLE_TESTING()

LIST(APPEND external_project_dependencies gtest)
