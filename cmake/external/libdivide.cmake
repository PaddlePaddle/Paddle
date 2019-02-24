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

set(WITH_LIBDIVIDE ON)
if(WIN32 OR APPLE)
    SET(WITH_LIBDIVIDE OFF CACHE STRING "Disable libdivide in Windows and MacOS" FORCE)
    return()
endif()

include(ExternalProject)

set(LIBDIVIDE_PROJECT       extern_libdivide)
set(LIBDIVIDE_PREFIX_DIR    ${THIRD_PARTY_PATH}/libdivide)
set(LIBDIVIDE_INSTALL_ROOT  ${THIRD_PARTY_PATH}/install/libdivide)
set(LIBDIVIDE_INC_DIR       ${LIBDIVIDE_INSTALL_ROOT}/include)

include_directories(${LIBDIVIDE_INC_DIR})

add_definitions(-DPADDLE_WITH_LIBDIVIDE)

ExternalProject_Add(
    ${LIBDIVIDE_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ""
    GIT_REPOSITORY      "https://github.com/ridiculousfish/libdivide.git"
    GIT_TAG             "v1.0"  # Jan,2018
    PREFIX              ${LIBDIVIDE_PREFIX_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${LIBDIVIDE_INSTALL_ROOT}
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${LIBDIVIDE_INSTALL_ROOT}
)

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/libdivide_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_libdivide = \"${dummyfile}\";")
    add_library(libdivide STATIC ${dummyfile})
else()
    add_library(libdivide INTERFACE)
endif()

add_dependencies(libdivide ${LIBDIVIDE_PROJECT})
