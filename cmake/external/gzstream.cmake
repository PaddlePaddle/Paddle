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
#

include (ExternalProject)

# NOTE: gzstream is needed when linking with ctr reader.

SET(GZSTREAM_SOURCES_DIR ${THIRD_PARTY_PATH}/gzstream)
SET(GZSTREAM_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gzstream)
SET(GZSTREAM_INCLUDE_DIR "${GZSTREAM_INSTALL_DIR}/include/" CACHE PATH "gzstream include directory." FORCE)

ExternalProject_Add(
        extern_gzstream
        DEPENDS zlib
        GIT_REPOSITORY "https://github.com/jacquesqiao/gzstream.git"
        GIT_TAG ""
        PREFIX          ${GZSTREAM_SOURCES_DIR}
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE 1
        BUILD_COMMAND   make EXTERN_CPPFLAGS="-I${THIRD_PARTY_PATH}/install/zlib/include" EXTERM_LDFLAGS="-L${THIRD_PARTY_PATH}/install/zlib/lib" -j8
        INSTALL_COMMAND mkdir -p ${GZSTREAM_INSTALL_DIR}/lib/ && mkdir -p ${GZSTREAM_INSTALL_DIR}/include/
        && cp ${GZSTREAM_SOURCES_DIR}/src/extern_gzstream/libgzstream.a ${GZSTREAM_INSTALL_DIR}/lib
        && cp -r ${GZSTREAM_SOURCES_DIR}/src/extern_gzstream/gzstream.h ${GZSTREAM_INSTALL_DIR}/include
)

ADD_LIBRARY(gzstream STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gzstream PROPERTY IMPORTED_LOCATION
        "${GZSTREAM_INSTALL_DIR}/lib/libgzstream.a")

include_directories(${GZSTREAM_INCLUDE_DIR})
ADD_DEPENDENCIES(gzstream extern_gzstream zlib)
