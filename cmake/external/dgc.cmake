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

INCLUDE(ExternalProject)

SET(DGC_SOURCES_DIR "${THIRD_PARTY_PATH}/dgc")
SET(DGC_INSTALL_DIR "${THIRD_PARTY_PATH}/install/dgc")
SET(DGC_INCLUDE_DIR "${DGC_INSTALL_DIR}/include" CACHE PATH "dgc include directory." FORCE)
SET(DGC_LIBRARIES "${DGC_INSTALL_DIR}/lib/libdgc.a" CACHE FILEPATH "dgc library." FORCE)
INCLUDE_DIRECTORIES(${DGC_INCLUDE_DIR})

ExternalProject_Add(
    extern_dgc
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY "https://github.com/PaddlePaddle/Fleet"
    GIT_TAG "2d04dc3800cdd0601f1b65d547dabcc60b0cf9dc"
    SOURCE_DIR "${DGC_SOURCES_DIR}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd collective && make -j
    INSTALL_COMMAND mkdir -p ${DGC_INSTALL_DIR}/lib/  ${DGC_INCLUDE_DIR}/dgc
        && cp ${DGC_SOURCES_DIR}/collective/build/lib/libdgc.a ${DGC_LIBRARIES}
        && cp ${DGC_SOURCES_DIR}/collective/build/include/dgc.h ${DGC_INCLUDE_DIR}/dgc/
    BUILD_IN_SOURCE 1
)

ADD_LIBRARY(dgc STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET dgc PROPERTY IMPORTED_LOCATION ${DGC_LIBRARIES})
ADD_DEPENDENCIES(dgc extern_dgc)

