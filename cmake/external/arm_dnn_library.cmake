# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT WITH_ARM_DNN_LIBRARY)
  return()
endif()

if(NOT ARM_DNN_LIBRARY_REPOSITORY)
  # Arm DNN library is maintained as a standalone library in the Paddle Lite repository.
  set(ARM_DNN_LIBRARY_REPOSITORY ${GIT_URL}/PaddlePaddle/Paddle-Lite.git)
endif()

if(NOT ARM_DNN_LIBRARY_TAG)
  set(ARM_DNN_LIBRARY_TAG develop)
endif()

if(NOT ARM_DNN_LIBRARY_SOURCE_SUBDIR)
  set(ARM_DNN_LIBRARY_SOURCE_SUBDIR lite/backends/arm/arm_dnn_library)
endif()

message(STATUS "Arm DNN Library repository: " ${ARM_DNN_LIBRARY_REPOSITORY})
message(STATUS "Arm DNN Library tag: " ${ARM_DNN_LIBRARY_TAG})
message(STATUS "Arm DNN Library dir: " ${ARM_DNN_LIBRARY_DIR})

include(ExternalProject)

set(ARM_DNN_LIBRARY_PROJECT "extern_arm_dnn_library")
set(ARM_DNN_LIBRARY_PREFIX_DIR ${THIRD_PARTY_PATH}/arm_dnn_library)
set(ARM_DNN_LIBRARY_SOURCE_DIR
    ${ARM_DNN_LIBRARY_PREFIX_DIR}/src/${ARM_DNN_LIBRARY_PROJECT}/${ARM_DNN_LIBRARY_SOURCE_SUBDIR}
)
set(ARM_DNN_LIBRARY_INSTALL_DIR ${THIRD_PARTY_PATH}/install/arm_dnn_library)
set(ARM_DNN_LIBRARY_INCLUDE_DIR
    "${ARM_DNN_LIBRARY_INSTALL_DIR}/include"
    "${ARM_DNN_LIBRARY_SOURCE_DIR}/include"
    CACHE PATH "arm dnn library include directory." FORCE)
set(ARM_DNN_LIBRARY_LIBRARY_DIR
    "${ARM_DNN_LIBRARY_INSTALL_DIR}/lib"
    CACHE PATH "arm dnn library library directory." FORCE)
if(WIN32)
  set(ARM_DNN_LIBRARY_BINARY_PATH
      "${ARM_DNN_LIBRARY_LIBRARY_DIR}/libarm_dnn_library.lib"
      CACHE FILEPATH "arm dnn library library path." FORCE)
else()
  set(ARM_DNN_LIBRARY_BINARY_PATH
      "${ARM_DNN_LIBRARY_LIBRARY_DIR}/libarm_dnn_library.a"
      CACHE FILEPATH "arm dnn library library path." FORCE)
endif()
set(ARM_DNN_LIBRARY_OPTIONAL_ARGS
    -DARM_DNN_LIBRARY_LIBRARY_TYPE=static
    -DARM_DNN_LIBRARY_WITH_THREAD_POOL=ON
    -DCMAKE_INSTALL_PREFIX=${ARM_DNN_LIBRARY_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE})
set(ARM_DNN_LIBRARY_BUILD_COMMAND ${CMAKE_COMMAND} --build . -j)

include_directories(${ARM_DNN_LIBRARY_INCLUDE_DIR})

ExternalProject_Add(
  ${ARM_DNN_LIBRARY_PROJECT}
  GIT_REPOSITORY ${ARM_DNN_LIBRARY_REPOSITORY}
  GIT_TAG ${ARM_DNN_LIBRARY_TAG}
  GIT_SUBMODULES "" GIT_SUBMODULES_RECURSE OFF
  PREFIX ${ARM_DNN_LIBRARY_PREFIX_DIR}
  SOURCE_SUBDIR ${ARM_DNN_LIBRARY_SOURCE_SUBDIR}
  BUILD_COMMAND ${ARM_DNN_LIBRARY_BUILD_COMMAND}
  CMAKE_ARGS ${ARM_DNN_LIBRARY_OPTIONAL_ARGS}}
  BUILD_BYPRODUCTS ${ARM_DNN_LIBRARY_BINARY_PATH})

add_library(arm_dnn_library STATIC IMPORTED GLOBAL)
set_property(TARGET arm_dnn_library PROPERTY IMPORTED_LOCATION
                                             ${ARM_DNN_LIBRARY_BINARY_PATH})
add_dependencies(arm_dnn_library ${ARM_DNN_LIBRARY_PROJECT})
