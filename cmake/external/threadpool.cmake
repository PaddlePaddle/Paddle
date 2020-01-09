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

SET(THREADPOOL_PREFIX_DIR ${THIRD_PARTY_PATH}/threadpool)
SET(THREADPOOL_SOURCE_DIR ${THIRD_PARTY_PATH}/threadpool/src/extern_threadpool)
SET(THREADPOOL_REPOSITORY https://github.com/progschj/ThreadPool.git)
SET(THREADPOOL_TAG        9a42ec1329f259a5f4881a291db1dcb8f2ad9040)

cache_third_party(extern_threadpool
    REPOSITORY   ${THREADPOOL_REPOSITORY}
    TAG          ${THREADPOOL_TAG}
    DIR          THREADPOOL_SOURCE_DIR)

SET(THREADPOOL_INCLUDE_DIR ${THREADPOOL_SOURCE_DIR})
INCLUDE_DIRECTORIES(${THREADPOOL_INCLUDE_DIR})

ExternalProject_Add(
    extern_threadpool
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${THREADPOOL_DOWNLOAD_CMD}"
    PREFIX          ${THREADPOOL_PREFIX_DIR}
    SOURCE_DIR      ${THREADPOOL_SOURCE_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/threadpool_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_threadpool = \"${dummyfile}\";")
    add_library(simple_threadpool STATIC ${dummyfile})
else()
    add_library(simple_threadpool INTERFACE)
endif()

add_dependencies(simple_threadpool extern_threadpool)
