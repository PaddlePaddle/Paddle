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

set(ROCKSDB_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/rocksdb)
set(ROCKSDB_TAG 6.19.fb)

set(JEMALLOC_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/jemalloc/include)
set(JEMALLOC_LIBRARIES
    ${THIRD_PARTY_PATH}/install/jemalloc/lib/libjemalloc_pic.a)
message(STATUS "rocksdb jemalloc:" ${JEMALLOC_LIBRARIES})

set(ROCKSDB_PREFIX_DIR ${THIRD_PARTY_PATH}/rocksdb)
set(ROCKSDB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/rocksdb)
set(ROCKSDB_INCLUDE_DIR
    "${ROCKSDB_INSTALL_DIR}/include"
    CACHE PATH "rocksdb include directory." FORCE)
set(ROCKSDB_LIBRARIES
    "${ROCKSDB_INSTALL_DIR}/lib/librocksdb.a"
    CACHE FILEPATH "rocksdb library." FORCE)

set(ROCKSDB_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DROCKSDB_LIBAIO_PRESENT -I${JEMALLOC_INCLUDE_DIR}")
set(ROCKSDB_SHARED_LINKER_FLAGS "-Wl,--no-as-needed -ldl")

if(WITH_ARM)
  file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/rocksdb/libaio.h.patch
       libaio_h_patch)
  set(ROCKSDB_PATCH_COMMAND_ARM
      patch -Nd ${PADDLE_SOURCE_DIR}/third_party/rocksdb/env/ <
      ${libaio_h_patch})
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION}
                                            VERSION_GREATER_EQUAL 13.0)
  message(
    STATUS
      "GCC version is ${CMAKE_CXX_COMPILER_VERSION}, adding some patches for gcc13+"
  )
  file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/rocksdb/include.patch
       include_patch)
  file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/rocksdb/redundant_move.patch
       redundant_move_patch)
  set(ROCKSDB_PATCH_COMMAND_GCC13 git apply ${include_patch}
                                  ${redundant_move_patch})
endif()

ExternalProject_Add(
  extern_rocksdb
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${ROCKSDB_PREFIX_DIR}
  SOURCE_DIR ${ROCKSDB_SOURCE_DIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND
  COMMAND git checkout -- . && git checkout ${ROCKSDB_TAG}
  COMMAND ${ROCKSDB_PATCH_COMMAND_ARM}
  COMMAND ${ROCKSDB_PATCH_COMMAND_GCC13}
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DWITH_BZ2=OFF
             -DPORTABLE=1
             -DWITH_GFLAGS=OFF
             -DWITH_TESTS=OFF
             -DWITH_JEMALLOC=ON
             -DJeMalloc_LIBRARIES=${JEMALLOC_LIBRARIES}
             -DJeMalloc_INCLUDE_DIRS=${JEMALLOC_INCLUDE_DIR}
             -DWITH_BENCHMARK_TOOLS=OFF
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_CXX_FLAGS=${ROCKSDB_CXX_FLAGS}
             -DCMAKE_SHARED_LINKER_FLAGS=${ROCKSDB_SHARED_LINKER_FLAGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${ROCKSDB_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR:PATH=${ROCKSDB_INSTALL_DIR}/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${ROCKSDB_LIBRARIES})
add_dependencies(extern_rocksdb snappy extern_jemalloc)

add_library(rocksdb STATIC IMPORTED GLOBAL)
set_property(TARGET rocksdb PROPERTY IMPORTED_LOCATION ${ROCKSDB_LIBRARIES})
include_directories(${ROCKSDB_INCLUDE_DIR})
add_dependencies(rocksdb extern_rocksdb)

list(APPEND external_project_dependencies rocksdb)
