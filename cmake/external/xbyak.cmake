# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
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

set(WITH_XBYAK ON)
if(WIN32 OR APPLE)
    SET(WITH_XBYAK OFF CACHE STRING "Disable XBYAK in Windows and MacOS" FORCE)
    return()
endif()

include(ExternalProject)

set(XBYAK_PROJECT       extern_xbyak)
set(XBYAK_PREFIX_DIR    ${THIRD_PARTY_PATH}/xbyak)
set(XBYAK_INSTALL_ROOT  ${THIRD_PARTY_PATH}/install/xbyak)
set(XBYAK_INC_DIR       ${XBYAK_INSTALL_ROOT}/include)

include_directories(${XBYAK_INC_DIR})
include_directories(${XBYAK_INC_DIR}/xbyak)

add_definitions(-DPADDLE_WITH_XBYAK)

# xbyak options
add_definitions(-DXBYAK64)
add_definitions(-DXBYAK_NO_OP_NAMES)

ExternalProject_Add(
    ${XBYAK_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ""
    GIT_REPOSITORY      "https://github.com/herumi/xbyak.git"
    GIT_TAG             "v5.661"  # Jul 26th
    PREFIX              ${XBYAK_PREFIX_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${XBYAK_INSTALL_ROOT}
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${XBYAK_INSTALL_ROOT}
)

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/xbyak_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_xbyak = \"${dummyfile}\";")
    add_library(xbyak STATIC ${dummyfile})
else()
    add_library(xbyak INTERFACE)
endif()

add_dependencies(xbyak ${XBYAK_PROJECT})
