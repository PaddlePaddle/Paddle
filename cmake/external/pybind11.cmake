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
SET(PYBIND_REPOSITORY     ${GIT_URL}/pybind/pybind11.git)
SET(PYBIND_TAG            v2.4.3)

set(PYBIND_INCLUDE_DIR ${THIRD_PARTY_PATH}/pybind/src/extern_pybind/include)
include_directories(${PYBIND_INCLUDE_DIR})

ExternalProject_Add(
        extern_pybind
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        GIT_REPOSITORY    ${PYBIND_REPOSITORY}
        GIT_TAG           ${PYBIND_TAG}
        PREFIX            ${PYBIND_PREFIX_DIR}
        # If we explicitly leave the `UPDATE_COMMAND` of the ExternalProject_Add
        # function in CMakeLists blank, it will cause another parameter GIT_TAG
        # to be modified without triggering incremental compilation, and the
        # third-party library version changes cannot be incorporated.
        # reference: https://cmake.org/cmake/help/latest/module/ExternalProject.html
        UPDATE_COMMAND    ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
)

add_library(pybind INTERFACE)

add_dependencies(pybind extern_pybind)
