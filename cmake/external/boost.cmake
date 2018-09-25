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

set(BOOST_PROJECT       "extern_boost")
# To release PaddlePaddle as a pip package, we have to follow the
# manylinux1 standard, which features as old Linux kernels and
# compilers as possible and recommends CentOS 5. Indeed, the earliest
# CentOS version that works with NVIDIA CUDA is CentOS 6.  And a new
# version of boost, say, 1.66.0, doesn't build on CentOS 6.  We
# checked that the devtools package of CentOS 6 installs boost 1.41.0.
# So we use 1.41.0 here.
set(BOOST_VER           "1.41.0")
if((NOT DEFINED BOOST_TAR) OR (NOT DEFINED BOOST_URL))
    message(STATUS "use pre defined download url")
    set(BOOST_TAR "boost_1_41_0" CACHE STRING "" FORCE)
    set(BOOST_URL "http://paddlepaddledeps.cdn.bcebos.com/${BOOST_TAR}.tar.gz" CACHE STRING "" FORCE)
endif()
IF (WIN32)
    MESSAGE(WARNING, "In windows, boost can not be downloaded automaticlly, please build it manually and put it at " ${THIRD_PARTY_PATH}install/boost)
else()
    MESSAGE(STATUS "BOOST_TAR: ${BOOST_TAR}, BOOST_URL: ${BOOST_URL}")
ENDIF(WIN32)

set(BOOST_SOURCES_DIR ${THIRD_PARTY_PATH}/boost)
set(BOOST_DOWNLOAD_DIR  "${BOOST_SOURCES_DIR}/src/${BOOST_PROJECT}")
set(BOOST_INCLUDE_DIR "${BOOST_DOWNLOAD_DIR}/${BOOST_TAR}" CACHE PATH "boost include directory." FORCE)
set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

include_directories(${BOOST_INCLUDE_DIR})

if (NOT WIN32)
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
endif(NOT WIN32)

if (${CMAKE_VERSION} VERSION_LESS "3.3.0" OR NOT WIN32)
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/boost_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
    add_library(boost STATIC ${dummyfile})
else()
    add_library(boost INTERFACE)
endif()

add_dependencies(boost ${BOOST_PROJECT})
list(APPEND external_project_dependencies boost)
set(Boost_INCLUDE_DIR ${BOOST_INCLUDE_DIR})
