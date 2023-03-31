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

include(ExternalProject)

set(TARGET_NAME "extern_k2")
set(K2_PATH
    "${THIRD_PARTY_PATH}/k2"
    CACHE STRING "A path setting for external_k2 path.")
set(K2_PREFIX_DIR ${K2_PATH})
set(K2_INSTALL_DIR
    ${THIRD_PARTY_PATH}/install/k2
    CACHE STRING "A path setting for external_k2 install path.")

# set(K2_REPOSITORY https://github.com/k2-fsa/k2.git)
# set(K2_TAG v1.23.4)

set(K2_REPOSITORY https://github.com/zh794390558/k2.git)
set(K2_TAG wo_pytorch)

set(K2_INCLUDE_DIR ${K2_PREFIX_DIR}/src/${TARGET_NAME})
set(K2_BUILD_DIR ${K2_PREFIX_DIR}/src/${TARGET_NAME}-build)
set(K2_LIBRARIES_DIR ${K2_INSTALL_DIR}/lib)
if(WIN32)
  set(K2_LIBRARIES
      "${K2_LIBRARIES_DIR}/k2context${CMAKE_SHARED_LIBRARY_SUFFIX}"
      "${K2_LIBRARIES_DIR}/k2fsa${CMAKE_SHARED_LIBRARY_SUFFIX}"
      "${K2_LIBRARIES_DIR}/k2_log${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "k2 context fsa log Library" FORCE)
else()
  set(K2_LIBRARIES
      "${K2_LIBRARIES_DIR}/libk2context${CMAKE_SHARED_LIBRARY_SUFFIX}"
      "${K2_LIBRARIES_DIR}/libk2fsa${CMAKE_SHARED_LIBRARY_SUFFIX}"
      "${K2_LIBRARIES_DIR}/libk2_log${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "k2 context fsa log Library" FORCE)
endif()


set(K2_ENABLE_NVTX OFF)

if(WITH_GPU)
  set(K2_WITH_CUDA ON)
  if(NOT WIN32)
    set(K2_ENABLE_NVTX ON)
  endif()
else()
  set(K2_WITH_CUDA OFF)
endif()
message(STATUS "K2_WITH_CUDA=${K2_WITH_CUDA}")

if(WIN32)
  set(K2_BUILD_LIB_DIR ${K2_BUILD_DIR}/bin)
else()
  set(K2_BUILD_LIB_DIR ${K2_BUILD_DIR}/lib)
endif()

ExternalProject_Add(
  ${TARGET_NAME}
  ${EXTERNAL_PROJECT_LOG_ARGS} ${SHALLOW_CLONE}
  GIT_REPOSITORY ${K2_REPOSITORY}
  GIT_TAG ${K2_TAG}
  PREFIX ${K2_PREFIX_DIR}
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DBUILD_SHARED_LIBS=ON
             -DK2_WITH_CUDA=${K2_WITH_CUDA}
             -DK2_ENABLE_NVTX=${K2_ENABLE_NVTX}
             -DK2_USE_PYTORCH=OFF
             -DK2_ENABLE_TESTS=OFF
             -DK2_ENABLE_BENCHMARK=OFF
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
                   -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
  INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${K2_BUILD_LIB_DIR}
                  ${K2_LIBRARIES_DIR}
  BUILD_BYPRODUCTS ${K2_LIBRARIES})

add_library(k2 INTERFACE)
message("K2_INSTALL_DIR=${K2_INSTALL_DIR}")
message("K2_INCLUDE_DIR=${K2_INCLUDE_DIR}")
message("K2_LIBRARIES_DIR=${K2_LIBRARIES_DIR}")
message("K2_LIBRARIES=${K2_LIBRARIES}")
target_include_directories(k2 INTERFACE ${K2_INCLUDE_DIR})
target_link_libraries(k2 INTERFACE ${K2_LIBRARIES})
add_dependencies(k2 extern_k2)
