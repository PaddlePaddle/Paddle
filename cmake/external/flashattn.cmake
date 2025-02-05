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

if(WITH_ROCM)
  add_definitions(-DPADDLE_WITH_FLASHATTN)

  set(FA_REPOSITORY https://github.com/PaddlePaddle/flash-attention.git)
  set(FA_TAG "dcu")
  set(FLASHATTN_PREFIX_DIR ${THIRD_PARTY_PATH}/flashattn_hip)
  set(FLASHATTN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/flashattn)
  set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/flashattn_hip)

  set(FLASHATTN_INCLUDE_DIR
      "${FLASHATTN_INSTALL_DIR}/include"
      CACHE PATH "flash-attn Directory" FORCE)
  set(FLASHATTN_LIB_DIR
      "${FLASHATTN_INSTALL_DIR}/lib"
      CACHE PATH "flash-attn Library Directory" FORCE)
  set(FLASHATTN_LIBRARIES
      "${FLASHATTN_INSTALL_DIR}/lib/libflashattn${CMAKE_SHARED_LIBRARY_SUFFIX}"
      CACHE FILEPATH "flash-attn Library" FORCE)

  set(FLASHATTN_C_FLAGS ${CMAKE_C_FLAGS})
  set(FLASHATTN_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  set(FLASHATTN_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  set(FLASHATTN_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -w -Wno-deprecated-builtins -Wno-deprecated -DNDEBUG -U__HIP_NO_HALF_OPERATORS__ -U__HIP_NO_HALF_CONVERSIONS__ -fPIC -O3 -std=c++17 -D__HIP_PLATFORM_HCC__=1 --offload-arch=gfx928 -D__gfx940__ -mllvm -enable-num-vgprs-512=true"
  )
  set(FLASHATTN_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(FLASHATTN_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})

  ExternalProject_Add(
    extern_flashattn
    GIT_REPOSITORY ${FA_REPOSITORY}
    GIT_TAG ${FA_TAG}
    SOURCE_DIR ${SOURCE_DIR}
    PREFIX ${FLASHATTN_PREFIX_DIR}
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    #BUILD_ALWAYS    1
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc
               -DAMDGPU_TARGETS=gfx928
               -DCMAKE_CXX_COMPILER_LAUNCHER=${CCACHE_PATH}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${FLASHATTN_C_FLAGS}
               -DCMAKE_C_FLAGS_DEBUG=${FLASHATTN_C_FLAGS_DEBUG}
               -DCMAKE_C_FLAGS_RELEASE=${FLASHATTN_C_FLAGS_RELEASE}
               -DCMAKE_CXX_FLAGS=${FLASHATTN_CXX_FLAGS}
               -DCMAKE_CXX_FLAGS_RELEASE=${FLASHATTN_CXX_FLAGS_RELEASE}
               -DCMAKE_CXX_FLAGS_DEBUG=${FLASHATTN_CXX_FLAGS_DEBUG}
               -DCMAKE_INSTALL_PREFIX=${FLASHATTN_INSTALL_DIR}
               -DWITH_GPU=${WITH_GPU}
               -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
               -DWITH_ROCM=${WITH_ROCM}
               -DWITH_OMP=${USE_OMP}
               -DBUILD_SHARED=ON
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
               -DCMAKE_JOB_POOL_COMPILE:STRING=compile
               -DCMAKE_JOB_POOLS:STRING=compile=4
               ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_INSTALL_PREFIX:PATH=${FLASHATTN_INSTALL_DIR}
    BUILD_BYPRODUCTS ${FLASHATTN_LIBRARIES})
else()

  add_definitions(-DPADDLE_WITH_FLASHATTN)
  option(FA_BUILD_WITH_CACHE "Download cache so files from bos" ON)

  set(FLASHATTN_PREFIX_DIR ${THIRD_PARTY_PATH}/flashattn)
  set(FLASHATTN_SOURCE_SUBDIR csrc)
  set(FLASHATTN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/flashattn)
  set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/flashattn)

  # get FA git commit
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${SOURCE_DIR}
    OUTPUT_VARIABLE FLASHATTN_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "flashattn git commit: ${FLASHATTN_TAG}")

  set(FLASHATTN_INCLUDE_DIR
      "${FLASHATTN_INSTALL_DIR}/include"
      CACHE PATH "flash-attn Directory" FORCE)
  set(FLASHATTN_LIB_DIR
      "${FLASHATTN_INSTALL_DIR}/lib"
      CACHE PATH "flash-attn Library Directory" FORCE)

  if(WIN32)
    set(FLASHATTN_LIBRARIES
        "${FLASHATTN_INSTALL_DIR}/bin/flashattn${CMAKE_SHARED_LIBRARY_SUFFIX}"
        CACHE FILEPATH "flash-attn Library" FORCE)
    if(WITH_FLASHATTN_V3)
      set(FLASHATTN_V3_LIBRARIES
          "${FLASHATTN_INSTALL_DIR}/bin/libflashattnv3${CMAKE_SHARED_LIBRARY_SUFFIX}"
          CACHE FILEPATH "flash-attn Library" FORCE)
    endif()
  else()
    set(FLASHATTN_LIBRARIES
        "${FLASHATTN_INSTALL_DIR}/lib/libflashattn${CMAKE_SHARED_LIBRARY_SUFFIX}"
        CACHE FILEPATH "flash-attn Library" FORCE)
    if(WITH_FLASHATTN_V3)
      set(FLASHATTN_V3_LIBRARIES
          "${FLASHATTN_INSTALL_DIR}/lib/libflashattnv3${CMAKE_SHARED_LIBRARY_SUFFIX}"
          CACHE FILEPATH "flash-attn Library" FORCE)
    endif()
  endif()

  set(BUILD_BYPRODUCTS_LIST ${FLASHATTN_LIBRARIES})
  if(WITH_FLASHATTN_V3)
    add_definitions(-DPADDLE_WITH_FLASHATTN_V3)
    list(APPEND BUILD_BYPRODUCTS_LIST ${FLASHATTN_V3_LIBRARIES})
  endif()

  if(NOT DEFINED FA_JOB_POOLS_COMPILE)
    set(FA_JOB_POOLS_COMPILE 4)
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
     OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
     OR WIN32)
    set(USE_OMP OFF)
  else()
    set(USE_OMP ON)
  endif()

  if(WIN32)
    set(FLASHATTN_C_FLAGS $<FILTER:${CMAKE_C_FLAGS},EXCLUDE,/Zc:inline>)
    set(FLASHATTN_C_FLAGS_DEBUG
        $<FILTER:${CMAKE_C_FLAGS_DEBUG},EXCLUDE,/Zc:inline>)
    set(FLASHATTN_C_FLAGS_RELEASE
        $<FILTER:${CMAKE_C_FLAGS_RELEASE},EXCLUDE,/Zc:inline>)
    set(FLASHATTN_CXX_FLAGS $<FILTER:${CMAKE_CXX_FLAGS},EXCLUDE,/Zc:inline>)
    set(FLASHATTN_CXX_FLAGS_RELEASE
        $<FILTER:${CMAKE_CXX_FLAGS_RELEASE},EXCLUDE,/Zc:inline>)
    set(FLASHATTN_CXX_FLAGS_DEBUG
        $<FILTER:${CMAKE_CXX_FLAGS_DEBUG},EXCLUDE,/Zc:inline>)
  else()
    set(FLASHATTN_C_FLAGS ${CMAKE_C_FLAGS})
    set(FLASHATTN_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
    set(FLASHATTN_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
    set(FLASHATTN_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    set(FLASHATTN_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set(FLASHATTN_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  endif()

  set(FA_NVCC_ARCH_BIN "")
  foreach(arch ${NVCC_ARCH_BIN})
    string(STRIP ${arch} arch)
    if(arch STREQUAL "")
      continue()
    endif()

    if(FA_NVCC_ARCH_BIN STREQUAL "")
      set(FA_NVCC_ARCH_BIN "${arch}")
    else()
      set(FA_NVCC_ARCH_BIN "${FA_NVCC_ARCH_BIN}-${arch}")
    endif()
  endforeach()

  set(BASE_URL
      "https://xly-devops.bj.bcebos.com/gpups/flash-attention/cu${FA_NVCC_ARCH_BIN}"
  )
  set(TAR_FILE_NAME "flashattn_libs_${FLASHATTN_TAG}.tar")
  set(TAR_FILE_URL "${BASE_URL}/${TAR_FILE_NAME}")
  set(FA_BUILD_DIR "${FLASHATTN_PREFIX_DIR}/src/extern_flashattn-build/")
  set(CACHE_TAR_PATH "${FA_BUILD_DIR}/${TAR_FILE_NAME}")
  set(CACHE_TAR_DIR "${FA_BUILD_DIR}/flashattn_libs_${FLASHATTN_TAG}")

  set(SKIP_BUILD_FA OFF)
  if(FA_BUILD_WITH_CACHE)

    message(STATUS "Downloading ${TAR_FILE_URL} to ${CACHE_TAR_PATH}")
    file(
      DOWNLOAD "${TAR_FILE_URL}" "${CACHE_TAR_PATH}"
      STATUS DOWNLOAD_STATUS
      LOG DOWNLOAD_LOG)
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)

    if(DOWNLOAD_RESULT EQUAL 0)
      message(STATUS "Download Successful")

      file(MAKE_DIRECTORY ${FA_BUILD_DIR})

      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf ${CACHE_TAR_PATH}
        WORKING_DIRECTORY ${FA_BUILD_DIR}
        RESULT_VARIABLE TAR_RESULT)

      if(NOT TAR_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract ${CACHE_TAR_PATH}")
      endif()

      file(STRINGS ${CACHE_TAR_DIR}/MD5.txt FILE_MD5)

      # Strip any leading or trailing whitespace
      string(STRIP ${FILE_MD5} FILE_MD5)

      file(MD5 ${CACHE_TAR_DIR}/fa_libs.tar FILE_MD5_ACTUAL)

      message(STATUS "Expected MD5: ${FILE_MD5}")
      message(STATUS "Actual MD5:   ${FILE_MD5_ACTUAL}")

      if(NOT "${FILE_MD5}" STREQUAL "${FILE_MD5_ACTUAL}")
        message(
          FATAL_ERROR "MD5 checksum mismatch! The download may be corrupted.")
      else()
        message(STATUS "MD5 checksum verified successfully.")
      endif()

      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf ${CACHE_TAR_DIR}/fa_libs.tar
        WORKING_DIRECTORY ${CACHE_TAR_DIR}
        RESULT_VARIABLE TAR_RESULT)

      if(NOT TAR_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract ${CACHE_TAR_PATH}/fa_libs.tar")
      endif()

      file(GLOB_RECURSE SO_FILES "${CACHE_TAR_DIR}/fa_libs/*.so")
      foreach(so_file ${SO_FILES})
        message(STATUS "Copy ${so_file} to ${FA_BUILD_DIR}")
        message(STATUS "Copy ${so_file} to ${FLASHATTN_LIB_DIR}")
        file(COPY "${so_file}" DESTINATION "${FA_BUILD_DIR}")
        file(COPY "${so_file}" DESTINATION "${FLASHATTN_LIB_DIR}")
      endforeach()

      file(REMOVE_RECURSE ${CACHE_TAR_DIR})
      message(STATUS "Extraction completed in ${FA_BUILD_DIR}")

      set(SKIP_BUILD_FA ON)

    elseif(DOWNLOAD_RESULT EQUAL 6)
      message(
        STATUS
          "Could not resolve host. The given remote host was not resolvable.")
    elseif(DOWNLOAD_RESULT EQUAL 7)
      message(STATUS "Failed to connect to host.")
    elseif(DOWNLOAD_RESULT EQUAL 22)
      message(
        STATUS
          "HTTP page not retrieved. The requested URL was not found or a server returned a 4xx (client error) or 5xx (server error) response."
      )
    elseif(DOWNLOAD_RESULT EQUAL 28)
      message(
        STATUS
          "Operation timeout. The specified time-out period was reached according to the conditions."
      )
    else()
      message(STATUS "An error occurred. Error code: ${DOWNLOAD_RESULT}")
    endif()
  endif()

  ExternalProject_Add(
    extern_flashattn
    ${EXTERNAL_PROJECT_LOG_ARGS}
    SOURCE_DIR ${SOURCE_DIR}
    PREFIX ${FLASHATTN_PREFIX_DIR}
    SOURCE_SUBDIR ${FLASHATTN_SOURCE_SUBDIR}
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    #BUILD_ALWAYS    1
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${FLASHATTN_C_FLAGS}
               -DCMAKE_C_FLAGS_DEBUG=${FLASHATTN_C_FLAGS_DEBUG}
               -DCMAKE_C_FLAGS_RELEASE=${FLASHATTN_C_FLAGS_RELEASE}
               -DCMAKE_CXX_FLAGS=${FLASHATTN_CXX_FLAGS}
               -DCMAKE_CXX_FLAGS_RELEASE=${FLASHATTN_CXX_FLAGS_RELEASE}
               -DCMAKE_CXX_FLAGS_DEBUG=${FLASHATTN_CXX_FLAGS_DEBUG}
               -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}
               -DCMAKE_INSTALL_PREFIX=${FLASHATTN_INSTALL_DIR}
               -DWITH_GPU=${WITH_GPU}
               -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
               -DWITH_ROCM=${WITH_ROCM}
               -DWITH_OMP=${USE_OMP}
               -DBUILD_SHARED=ON
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
               -DCMAKE_JOB_POOL_COMPILE:STRING=compile
               -DCMAKE_JOB_POOLS:STRING=compile=${FA_JOB_POOLS_COMPILE}
               -DNVCC_ARCH_BIN=${FA_NVCC_ARCH_BIN}
               -DWITH_FLASHATTN_V3=${WITH_FLASHATTN_V3}
               -DSKIP_BUILD_FA=${SKIP_BUILD_FA}
               ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_INSTALL_PREFIX:PATH=${FLASHATTN_INSTALL_DIR}
    BUILD_BYPRODUCTS ${BUILD_BYPRODUCTS_LIST})
endif()

message(STATUS "flash-attn library: ${FLASHATTN_LIBRARIES}")
if(WITH_FLASHATTN_V3)
  message(STATUS "flash-attn-v3 library: ${FLASHATTN_V3_LIBRARIES}")
endif()
get_filename_component(FLASHATTN_LIBRARY_PATH ${FLASHATTN_LIBRARIES} DIRECTORY)
include_directories(${FLASHATTN_INCLUDE_DIR})

add_library(flashattn INTERFACE)
#set_property(TARGET flashattn PROPERTY IMPORTED_LOCATION ${FLASHATTN_LIBRARIES})
add_dependencies(flashattn extern_flashattn)
