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

if(NOT LINUX OR NOT WITH_MKL)
  message("Paddle-lite will not build because the required Linux and MKL do not exist.")
  set(WITH_LITE OFF)
  return()
endif()

if (NOT LITE_SOURCE_DIR OR NOT LITE_BINARY_DIR)
  include(ExternalProject)
  set(LITE_PROJECT extern_lite)
  set(LITE_SOURCES_DIR ${THIRD_PARTY_PATH}/lite)
  set(LITE_INSTALL_DIR ${THIRD_PARTY_PATH}/install/lite)

  if(NOT LITE_GIT_TAG)
    set(LITE_GIT_TAG 34c29406c27ee00cef033a98887403443eb2565f)
  endif()

  # No quotes, so cmake can resolve it as a command with arguments.
  set(LITE_BUILD_COMMAND $(MAKE) publish_inference -j)
  set(LITE_OPTIONAL_ARGS -DWITH_MKL=ON
                         -DLITE_WITH_CUDA=${WITH_GPU}
                         -DWITH_MKLDNN=OFF
                         -DLITE_WITH_X86=ON
                         -DLITE_WITH_PROFILE=OFF
                         -DWITH_LITE=OFF
                         -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF
                         -DWITH_PYTHON=OFF
                         -DWITH_TESTING=OFF
                         -DLITE_BUILD_EXTRA=ON
                         -DCUDNN_ROOT=${CUDNN_ROOT}
                         -DLITE_WITH_STATIC_CUDA=OFF
                         -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME}
                         -DLITE_WITH_ARM=OFF)

  ExternalProject_Add(
      ${LITE_PROJECT}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      GIT_REPOSITORY      "https://github.com/PaddlePaddle/Paddle-Lite.git"
      GIT_TAG             ${LITE_GIT_TAG}
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
  set(LITE_BINARY_DIR ${BINARY_DIR})
  set(LITE_SOURCE_DIR ${SOURCE_DIR})

endif()

message(STATUS "Paddle-lite BINARY_DIR: ${LITE_BINARY_DIR}")
message(STATUS "Paddle-lite SOURCE_DIR: ${LITE_SOURCE_DIR}")
include_directories(${LITE_SOURCE_DIR})
include_directories(${LITE_BINARY_DIR})

function(external_lite_static_libs alias path)
  add_library(${alias} SHARED IMPORTED GLOBAL)
  SET_PROPERTY(TARGET ${alias} PROPERTY IMPORTED_LOCATION
               ${path})
  if (LITE_PROJECT)
    add_dependencies(${alias} ${LITE_PROJECT})
  endif()
endfunction()

external_lite_static_libs(lite_full_static ${LITE_BINARY_DIR}/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so)

add_definitions(-DPADDLE_WITH_LITE)
add_definitions(-DLITE_WITH_LOG)
