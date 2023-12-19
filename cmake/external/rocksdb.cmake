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

# find_package(jemalloc REQUIRED)

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
set(ROCKSDB_COMMON_FLAGS
    "-g -pipe -O2 -W -Wall -Wno-unused-parameter -fPIC -fno-builtin-memcmp -fno-omit-frame-pointer"
)
set(ROCKSDB_FLAGS
    "-DNDEBUG -DROCKSDB_JEMALLOC -DJEMALLOC_NO_DEMANGLE -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DHAVE_PCLMUL -DZLIB -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_SUPPORT_THREAD_LOCAL -DROCKSDB_USE_RTTI -DROCKSDB_SCHED_GETCPU_PRESENT -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_AUXV_GETAUXVAL_PRESENT"
)
set(ROCKSDB_CMAKE_CXX_FLAGS
    "${ROCKSDB_COMMON_FLAGS} -DROCKSDB_LIBAIO_PRESENT ${ROCKSDB_FLAGS} -fPIC  -I${JEMALLOC_INCLUDE_DIR} -Wl,--no-as-needed -lz -ldl"
)
if(NOT WITH_ARM)
  set(ROCKSDB_FLAGS "${ROCKSDB_FLAGS} -DHAVE_SSE42")
  set(ROCKSDB_CMAKE_CXX_FLAGS
      "${ROCKSDB_CMAKE_CXX_FLAGS} -msse -msse4.2 -mpclmul")
endif()
set(ROCKSDB_CMAKE_C_FLAGS
    "${ROCKSDB_COMMON_FLAGS} ${ROCKSDB_FLAGS} -DROCKSDB_LIBAIO_PRESENT -fPIC  -I${JEMALLOC_INCLUDE_DIR}"
)
include_directories(${ROCKSDB_INCLUDE_DIR})

set(CMAKE_CXX_LINK_EXECUTABLE
    "${CMAKE_CXX_LINK_EXECUTABLE} -pthread -Wl,--no-as-needed -ldl -lrt -lz")
if(WITH_ARM)
  file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/rocksdb/libaio.h.patch
       native_src)
  set(ROCKSDB_PATCH_COMMAND
      git checkout -- . && git checkout ${ROCKSDB_TAG} && patch -Nd
      ${PADDLE_SOURCE_DIR}/third_party/rocksdb/env/ < ${native_src})
endif()
ExternalProject_Add(
  extern_rocksdb
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${ROCKSDB_PREFIX_DIR}
  SOURCE_DIR ${ROCKSDB_SOURCE_DIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND ${ROCKSDB_PATCH_COMMAND}
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DWITH_BZ2=OFF
             -DPORTABLE=1
             -DWITH_GFLAGS=OFF
             -DWITH_TESTS=OFF
             -DWITH_JEMALLOC=ON
             -DWITH_BENCHMARK_TOOLS=OFF
             -DJeMalloc_LIBRARIES=${JEMALLOC_LIBRARIES}
             -DJeMalloc_INCLUDE_DIRS=${JEMALLOC_INCLUDE_DIR}
             -DCMAKE_CXX_FLAGS=${ROCKSDB_CMAKE_CXX_FLAGS}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  INSTALL_COMMAND
    mkdir -p ${ROCKSDB_INSTALL_DIR}/lib/ && cp
    ${ROCKSDB_PREFIX_DIR}/src/extern_rocksdb-build/librocksdb.a
    ${ROCKSDB_LIBRARIES} && cp -r ${ROCKSDB_SOURCE_DIR}/include
    ${ROCKSDB_INSTALL_DIR}/
  BUILD_BYPRODUCTS ${ROCKSDB_LIBRARIES})

add_library(rocksdb STATIC IMPORTED GLOBAL)

add_dependencies(extern_rocksdb snappy)
add_dependencies(extern_rocksdb extern_jemalloc)
set_property(TARGET rocksdb PROPERTY IMPORTED_LOCATION ${ROCKSDB_LIBRARIES})
add_dependencies(rocksdb extern_rocksdb)

list(APPEND external_project_dependencies rocksdb)
