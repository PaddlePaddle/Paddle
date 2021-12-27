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

include(ExternalProject)

set(CONCURRENTQUEUE_PROJECT "extern_concurrentqueue")
set(CONCURRENTQUEUE_VER "v1.0.3")
SET(CONCURRENTQUEUE_URL_MD5 118e5bb661b567634647312991e10222)
set(CONCURRENTQUEUE_PREFIX_URL "https://github.com/cameron314/concurrentqueue/archive/refs/tags")
set(CONCURRENTQUEUE_URL "${CONCURRENTQUEUE_PREFIX_URL}/${CONCURRENTQUEUE_VER}.tar.gz")

MESSAGE(STATUS "CONCURRENTQUEUE_VERSION: ${CONCURRENTQUEUE_VER}, CONCURRENTQUEUE_URL: ${CONCURRENTQUEUE_URL}")

set(CONCURRENTQUEUE_PREFIX_DIR ${THIRD_PARTY_PATH}/concurrentqueue)
set(CONCURRENTQUEUE_SOURCE_DIR ${THIRD_PARTY_PATH}/concurrentqueue/src/)
set(CONCURRENTQUEUE_INCLUDE_DIR "${CONCURRENTQUEUE_SOURCE_DIR}/extern_concurrentqueue")

ExternalProject_Add(
    ${CONCURRENTQUEUE_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL                   ${CONCURRENTQUEUE_URL}
    URL_MD5               ${CONCURRENTQUEUE_URL_MD5}
    PREFIX                ${CONCURRENTQUEUE_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS  1
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    INSTALL_COMMAND       ""
    UPDATE_COMMAND        ""
    )

include_directories(${CONCURRENTQUEUE_INCLUDE_DIR})
