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

set(PYBIND_PREFIX_DIR     ${THIRD_PARTY_PATH}/pybind)
set(PYBIND_SOURCE_DIR     ${THIRD_PARTY_PATH}/pybind/src/extern_pybind)
SET(PYBIND_REPOSITORY     https://github.com/pybind/pybind11.git)
SET(PYBIND_TAG            v2.2.4)

cache_third_party(extern_pybind
    REPOSITORY    ${PYBIND_REPOSITORY}
    TAG           ${PYBIND_TAG}
    DIR           PYBIND_SOURCE_DIR)

set(PYBIND_INCLUDE_DIR ${PYBIND_SOURCE_DIR}/include)
include_directories(${PYBIND_INCLUDE_DIR})

ExternalProject_Add(
        extern_pybind
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        "${PYBIND_DOWNLOAD_CMD}"
        PREFIX            ${PYBIND_PREFIX_DIR}
        SOURCE_DIR        ${PYBIND_SOURCE_DIR}
        UPDATE_COMMAND    ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
)

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/pybind_dummy.c)
    file(WRITE ${dummyfile} "const char * dummy_pybind = \"${dummyfile}\";")
    add_library(pybind STATIC ${dummyfile})
else()
    add_library(pybind INTERFACE)
endif()

add_dependencies(pybind extern_pybind)
