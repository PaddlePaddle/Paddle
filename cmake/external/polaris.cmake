# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.
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

IF(NOT ${WITH_FPGA})
  return()
ENDIF(NOT ${WITH_FPGA})

INCLUDE(ExternalProject)

SET(POLARIS_PROJECT        "extern_polaris")
SET(POLARIS_SOURCES_DIR    ${THIRD_PARTY_PATH}/polaris)
SET(POLARIS_INSTALL_DIR    ${THIRD_PARTY_PATH}/install/polaris)
SET(POLARIS_INC_DIR        "${POLARIS_INSTALL_DIR}/include" CACHE PATH "polaris include directory." FORCE)

IF(WIN32 OR APPLE)
    MESSAGE(WARNING
        "Windows or Mac is not supported with baidu-FPGA in Paddle yet."
        "Force WITH_FPGA=OFF")
    SET(WITH_FPGA OFF CACHE STRING "Disable baidu-FPGA in Windows and MacOS" FORCE)
    return()
ENDIF()

SET(POLARIS_LIB "${POLARIS_INSTALL_DIR}/lib/libpolaris-sdk.a" CACHE FILEPATH "polaris library." FORCE)
MESSAGE(STATUS "Set ${POLARIS_INSTALL_DIR}/lib to runtime path")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${POLARIS_INSTALL_DIR}/lib")

INCLUDE_DIRECTORIES(${POLARIS_INC_DIR})

ExternalProject_Add(
    ${POLARIS_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY      https://github.com/baidu-fpga/Polaris.git
    PREFIX              ${POLARIS_SOURCES_DIR}
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       ""
    INSTALL_COMMAND     cp <SOURCE_DIR>/../extern_polaris ${THIRD_PARTY_PATH}/install/ -rf &&
                        mv ${THIRD_PARTY_PATH}/install/extern_polaris ${THIRD_PARTY_PATH}/install/polaris
    UPDATE_COMMAND      ""
)

ADD_LIBRARY(polaris STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET polaris PROPERTY IMPORTED_LOCATION ${POLARIS_LIB})
ADD_DEPENDENCIES(polaris ${POLARIS_PROJECT})

MESSAGE(STATUS "Polaris library: ${POLARIS_LIB}")

LIST(APPEND external_project_dependencies polaris)
