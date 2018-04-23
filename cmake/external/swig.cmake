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

IF(NOT WITH_SWIG_PY)
    return()
ENDIF()

FIND_PACKAGE(SWIG)

IF(NOT SWIG_FOUND)
    # build swig as an external project
    INCLUDE(ExternalProject)

    SET(SWIG_SOURCES_DIR ${THIRD_PARTY_PATH}/swig)
    SET(SWIG_INSTALL_DIR ${THIRD_PARTY_PATH}/install/swig)
    SET(SWIG_TARGET_VERSION "3.0.2")
    SET(SWIG_DOWNLOAD_SRC_MD5 "62f9b0d010cef36a13a010dc530d0d41")
    SET(SWIG_DOWNLOAD_WIN_MD5 "3f18de4fc09ab9abb0d3be37c11fbc8f")

    IF(WIN32)
        # swig.exe available as pre-built binary on Windows:
        ExternalProject_Add(swig
            URL                 http://prdownloads.sourceforge.net/swig/swigwin-${SWIG_TARGET_VERSION}.zip
            URL_MD5             ${SWIG_DOWNLOAD_WIN_MD5}
            SOURCE_DIR          ${SWIG_SOURCES_DIR}
            CONFIGURE_COMMAND   ""
            BUILD_COMMAND       ""
            INSTALL_COMMAND     ""
            UPDATE_COMMAND      ""
        )
        SET(SWIG_DIR ${SWIG_SOURCES_DIR} CACHE FILEPATH "SWIG Directory" FORCE)
        SET(SWIG_EXECUTABLE ${SWIG_SOURCES_DIR}/swig.exe  CACHE FILEPATH "SWIG Executable" FORCE)
    ELSE(WIN32)
        # swig uses bison find it by cmake and pass it down
        FIND_PACKAGE(BISON)

        # From SWIG configure
        ExternalProject_Add(swig
            GIT_REPOSITORY      https://github.com/swig/swig.git
            GIT_TAG             rel-3.0.10
            PREFIX              ${SWIG_SOURCES_DIR}
            CONFIGURE_COMMAND   cd <SOURCE_DIR> && ./autogen.sh && ./configure
                                --prefix=${SWIG_INSTALL_DIR} --without-pcre
            BUILD_COMMAND       cd <SOURCE_DIR> && make
            INSTALL_COMMAND     cd <SOURCE_DIR> && make install
            UPDATE_COMMAND      ""
        )

        SET(SWIG_DIR ${SWIG_INSTALL_DIR}/share/swig/${SWIG_TARGET_VERSION})
        SET(SWIG_EXECUTABLE ${SWIG_INSTALL_DIR}/bin/swig)
    ENDIF(WIN32)

    LIST(APPEND external_project_dependencies swig)
ENDIF(NOT SWIG_FOUND)
