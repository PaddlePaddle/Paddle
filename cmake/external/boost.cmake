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

include(ExternalProject)

set(BOOST_PROJECT       "extern_boost")
set(BOOST_VER           "1.41.0")
set(BOOST_TAR           "boost_1_41_0")
set(BOOST_URL           "http://sourceforge.net/projects/boost/files/boost/${BOOST_VER}/${BOOST_TAR}.tar.gz")
set(BOOST_SOURCES_DIR ${THIRD_PARTY_PATH}/boost)
set(BOOST_DOWNLOAD_DIR  "${BOOST_SOURCES_DIR}/src/${BOOST_PROJECT}")
set(BOOST_INCLUDE_DIR "${BOOST_DOWNLOAD_DIR}/${BOOST_TAR}" CACHE PATH "boost include directory." FORCE)
set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

include_directories(${BOOST_INCLUDE_DIR})

ExternalProject_Add(
    ${BOOST_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DOWNLOAD_DIR          ${BOOST_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${BOOST_URL} -c -q -O ${BOOST_TAR}.tar.gz
                          && tar zxf ${BOOST_TAR}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    PREFIX                ${BOOST_SOURCES_DIR}
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    INSTALL_COMMAND       ""
    UPDATE_COMMAND        ""
)

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/boost_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
    add_library(boost STATIC ${dummyfile})
else()
    add_library(boost INTERFACE)
endif()

add_dependencies(boost ${BOOST_PROJECT})
list(APPEND external_project_dependencies boost)
set(Boost_INCLUDE_DIR ${BOOST_INCLUDE_DIR})
