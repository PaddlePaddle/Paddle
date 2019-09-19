# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

set(LITE_PROJECT extern_lite)

set(LITE_SOURCES_DIR ${THIRD_PARTY_PATH}/lite)
set(LITE_INSTALL_DIR ${THIRD_PARTY_PATH}/install/lite)

# No quotes, so cmake can resolve it as a command with arguments.
set(LITE_BUILD_COMMAND $(MAKE) -j)
set(LITE_OPTIONAL_ARGS -DWITH_MKL=ON
                       -DWITH_GPU=OFF
                       -DWITH_MKLDNN=ON
                       -DLITE_WITH_X86=ON
                       -DLITE_WITH_PROFILE=OFF
                       -DWITH_LITE=ON
                       -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF
                       -DWITH_PYTHON=OFF
                       -DWITH_TESTING=ON
                       -DLITE_WITH_ARM=OFF)

ExternalProject_Add(
    ${LITE_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY      "https://github.com/Shixiaowei02/Paddle-Lite.git"
    GIT_TAG             "2a6362da32a2250ec7f7941b2263ad5ccd444241"
    PREFIX              ${LITE_SOURCES_DIR}
    UPDATE_COMMAND      ""
    BUILD_COMMAND       ${LITE_BUILD_COMMAND}
    INSTALL_COMMAND     ""
    CMAKE_ARGS          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_CXX_FLAGS=${LITE_CMAKE_CXX_FLAGS}
                        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                        -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                        -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                        ${EXTERNAL_OPTIONAL_ARGS}
                        ${LITE_OPTIONAL_ARGS}
)

ExternalProject_Get_property(${LITE_PROJECT} BINARY_DIR)
ExternalProject_Get_property(${LITE_PROJECT} SOURCE_DIR)

function(external_lite_static_libs alias path)
  add_library(${alias} STATIC IMPORTED GLOBAL) 
  SET_PROPERTY(TARGET ${alias} PROPERTY IMPORTED_LOCATION 
               ${path}) 
  add_dependencies(${alias} ${LITE_PROJECT})
endfunction()

include_directories(${SOURCE_DIR}/include)
external_lite_static_libs(api_full_static_1 ${BINARY_DIR}/lite/api/libapi_full_static.a)
