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

set(BOOST_PROJECT       "boost")
set(BOOST_VER           "1.66.0")
set(BOOST_TAR           "boost_1_66_0")
set(BOOST_URL           "https://dl.bintray.com/boostorg/release/${BOOST_VER}/source/${BOOST_TAR}.tar.gz")
set(BOOST_SOURCES_DIR ${THIRD_PARTY_PATH}/boost)
set(BOOST_DOWNLOAD_DIR  "${BOOST_SOURCES_DIR}/src/${BOOST_PROJECT}")
set(BOOST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/boost)
set(BOOST_ROOT ${BOOST_INSTALL_DIR} CACHE FILEPATH "boost root directory." FORCE)
set(BOOST_INCLUDE_DIR "${BOOST_INSTALL_DIR}/include" CACHE PATH "boost include directory." FORCE)

set(BOOST_BOOTSTRAP_CMD)
set(BOOST_B2_CMD)

if(UNIX)
  set(BOOST_BOOTSTRAP_CMD ./bootstrap.sh)
  set(BOOST_B2_CMD ./b2)
else()
  if(WIN32)
    set(BOOST_BOOTSTRAP_CMD bootstrap.bat)
    set(BOOST_B2_CMD b2.exe)
  endif()
endif()

INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIR})

ExternalProject_Add(
    ${BOOST_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DOWNLOAD_DIR          ${BOOST_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${BOOST_URL} -c -q -O ${BOOST_TAR}.tar.gz
                          && tar zxf ${BOOST_TAR}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    PREFIX                ${BOOST_SOURCES_DIR}
    CONFIGURE_COMMAND     sh -c "cd <SOURCE_DIR>/${BOOST_TAR} && ${BOOST_BOOTSTRAP_CMD} --prefix=${BOOST_INSTALL_DIR}"
    BUILD_COMMAND         cd <SOURCE_DIR>/${BOOST_TAR} && ${BOOST_B2_CMD} install --prefix=${BOOST_INSTALL_DIR} --with-program_options
    INSTALL_COMMAND       ""
    UPDATE_COMMAND        ""
)

ADD_DEPENDENCIES(boost ${BOOST_PROJECT})
LIST(APPEND external_project_dependencies boost)
if(WIN32)
  set(Boost_INCLUDE_DIR ${BOOST_INSTALL_DIR}/include/boost)
  set(BOOST_ROOT ${BOOST_INSTALL_DIR} )
else()
  set(Boost_INCLUDE_DIR ${BOOST_INSTALL_DIR}/include)
endif()

set(Boost_LIBRARY_DIR ${BOOST_INSTALL_DIR}/lib)
