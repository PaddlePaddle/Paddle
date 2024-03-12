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

include(ExternalProject)

add_definitions(-DPADDLE_WITH_FLASHATTN)

set(FLASHATTN_V1_PREFIX_DIR ${THIRD_PARTY_PATH}/flashattn_v1)
set(FLASHATTN_V1_SOURCE_SUBDIR csrc/flash_attn)
set(FLASHATTN_V1_INSTALL_DIR ${THIRD_PARTY_PATH}/install/flashattn_v1)
set(FLASHATTN_V1_REPOSITORY ${GIT_URL}/PaddlePaddle/flash-attention.git)
set(FLASHATTN_V1_TAG 5ff4bbf56ad066750407c4aef16ac740ebda0717)

set(FLASHATTN_V1_INCLUDE_DIR
    "${FLASHATTN_V1_INSTALL_DIR}/include"
    CACHE PATH "flash-attn v1 Directory" FORCE)
set(FLASHATTN_V1_LIB_DIR
    "${FLASHATTN_V1_INSTALL_DIR}/lib"
    CACHE PATH "flash-attn v1 Library Directory" FORCE)

if(WIN32)
  set(FLASHATTN_V1_OLD_LIBRARIES
      "${FLASHATTN_V1_INSTALL_DIR}/bin/flashattn${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "flash-attn v1 Library" FORCE)
  set(FLASHATTN_V1_LIBRARIES
      "${FLASHATTN_V1_INSTALL_DIR}/bin/flashattn_v1${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "flash-attn v1 Library" FORCE)
else()
  set(FLASHATTN_V1_OLD_LIBRARIES
      "${FLASHATTN_V1_INSTALL_DIR}/lib/libflashattn${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "flash-attn v1 Library" FORCE)
  set(FLASHATTN_V1_LIBRARIES
      "${FLASHATTN_V1_INSTALL_DIR}/lib/libflashattn_v1${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "flash-attn v1 Library" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
   OR WIN32)
  set(USE_OMP OFF)
else()
  set(USE_OMP ON)
endif()

if(WIN32)
  set(FLASHATTN_V1_C_FLAGS $<FILTER:${CMAKE_C_FLAGS},EXCLUDE,/Zc:inline>)
  set(FLASHATTN_V1_C_FLAGS_DEBUG
      $<FILTER:${CMAKE_C_FLAGS_DEBUG},EXCLUDE,/Zc:inline>)
  set(FLASHATTN_V1_C_FLAGS_RELEASE
      $<FILTER:${CMAKE_C_FLAGS_RELEASE},EXCLUDE,/Zc:inline>)
  set(FLASHATTN_V1_CXX_FLAGS $<FILTER:${CMAKE_CXX_FLAGS},EXCLUDE,/Zc:inline>)
  set(FLASHATTN_V1_CXX_FLAGS_RELEASE
      $<FILTER:${CMAKE_CXX_FLAGS_RELEASE},EXCLUDE,/Zc:inline>)
  set(FLASHATTN_V1_CXX_FLAGS_DEBUG
      $<FILTER:${CMAKE_CXX_FLAGS_DEBUG},EXCLUDE,/Zc:inline>)
else()
  set(FLASHATTN_V1_C_FLAGS ${CMAKE_C_FLAGS})
  set(FLASHATTN_V1_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  set(FLASHATTN_V1_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  set(FLASHATTN_V1_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(FLASHATTN_V1_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(FLASHATTN_V1_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()

ExternalProject_Add(
  extern_flashattn_v1
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${FLASHATTN_V1_REPOSITORY}
  GIT_TAG ${FLASHATTN_V1_TAG}
  PREFIX ${FLASHATTN_V1_PREFIX_DIR}
  SOURCE_SUBDIR ${FLASHATTN_V1_SOURCE_SUBDIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  #BUILD_ALWAYS    1
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${FLASHATTN_V1_C_FLAGS}
             -DCMAKE_C_FLAGS_DEBUG=${FLASHATTN_V1_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${FLASHATTN_V1_C_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS=${FLASHATTN_V1_CXX_FLAGS}
             -DCMAKE_CXX_FLAGS_RELEASE=${FLASHATTN_V1_CXX_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${FLASHATTN_V1_CXX_FLAGS_DEBUG}
             -DCMAKE_INSTALL_PREFIX=${FLASHATTN_V1_INSTALL_DIR}
             -DWITH_GPU=${WITH_GPU}
             -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
             -DWITH_ROCM=${WITH_ROCM}
             -DWITH_OMP=${USE_OMP}
             -DBUILD_SHARED=ON
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${FLASHATTN_V1_INSTALL_DIR}
  BUILD_BYPRODUCTS ${FLASHATTN_V1_LIBRARIES})

add_custom_target(
  extern_flashattn_v1_move_lib
  COMMAND ${CMAKE_COMMAND} -E copy ${FLASHATTN_V1_OLD_LIBRARIES}
          ${FLASHATTN_V1_LIBRARIES})

add_dependencies(extern_flashattn_v1_move_lib extern_flashattn_v1)

message(STATUS "flash-attn v1 library: ${FLASHATTN_V1_LIBRARIES}")
get_filename_component(FLASHATTN_V1_LIBRARY_PATH ${FLASHATTN_V1_LIBRARIES}
                       DIRECTORY)
include_directories(${FLASHATTN_V1_INCLUDE_DIR})

add_library(flashattn_v1 INTERFACE)
#set_property(TARGET flashattn_v1 PROPERTY IMPORTED_LOCATION ${FLASHATTN_V1_LIBRARIES})
add_dependencies(flashattn_v1 extern_flashattn_v1 extern_flashattn_v1_move_lib)
