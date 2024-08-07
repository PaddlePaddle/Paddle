# Copyright (c) 2017-2023 PaddlePaddle Authors. All Rights Reserved.
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

set(ONEDNN_PROJECT "extern_onednn")
set(ONEDNN_PREFIX_DIR ${THIRD_PARTY_PATH}/onednn)
set(ONEDNN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/onednn)
set(ONEDNN_INC_DIR
    "${ONEDNN_INSTALL_DIR}/include"
    CACHE PATH "oneDNN include directory." FORCE)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/onednn)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)
set(LIBDIR "lib")
if(CMAKE_INSTALL_LIBDIR MATCHES ".*lib64$")
  set(LIBDIR "lib64")
endif()

message(STATUS "Set ${ONEDNN_INSTALL_DIR}/${LIBDIR} to runtime path")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
                        "${ONEDNN_INSTALL_DIR}/${LIBDIR}")

include_directories(${ONEDNN_INC_DIR}
)# For oneDNN code to include internal headers.

if(NOT WIN32)
  set(ONEDNN_FLAG
      "-Wno-error=strict-overflow -Wno-error=unused-result -Wno-error=array-bounds"
  )
  set(ONEDNN_FLAG "${ONEDNN_FLAG} -Wno-unused-result -Wno-unused-value")
  set(ONEDNN_CFLAG "${CMAKE_C_FLAGS} ${ONEDNN_FLAG}")
  set(ONEDNN_CXXFLAG "${CMAKE_CXX_FLAGS} ${ONEDNN_FLAG}")
  set(ONEDNN_CXXFLAG_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  set(ONEDNN_CFLAG_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
  set(ONEDNN_LIB
      "${ONEDNN_INSTALL_DIR}/${LIBDIR}/libdnnl.so"
      CACHE FILEPATH "oneDNN library." FORCE)
else()
  set(ONEDNN_CXXFLAG "${CMAKE_CXX_FLAGS} /EHsc")
  set(ONEDNN_CFLAG "${CMAKE_C_FLAGS}")
  string(REPLACE "/O2 " "" ONEDNN_CFLAG_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
  string(REPLACE "/O2 " "" ONEDNN_CXXFLAG_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  set(ONEDNN_LIB
      "${ONEDNN_INSTALL_DIR}/bin/mkldnn.lib"
      CACHE FILEPATH "oneDNN library." FORCE)
endif()

if(LINUX)
  set(BUILD_BYPRODUCTS_ARGS ${ONEDNN_LIB})
else()
  set(BUILD_BYPRODUCTS_ARGS "")
endif()

ExternalProject_Add(
  ${ONEDNN_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  DEPENDS ${ONEDNN_DEPENDS}
  PREFIX ${ONEDNN_PREFIX_DIR}
  UPDATE_COMMAND ""
  #BUILD_ALWAYS        1
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_FLAGS=${ONEDNN_CXXFLAG}
             -DCMAKE_CXX_FLAGS_RELEASE=${ONEDNN_CXXFLAG_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS=${ONEDNN_CFLAG}
             -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${ONEDNN_CFLAG_RELEASE}
             -DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DDNNL_BUILD_TESTS=OFF
             -DDNNL_BUILD_EXAMPLES=OFF
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ONEDNN_INSTALL_DIR}
  BUILD_BYPRODUCTS ${BUILD_BYPRODUCTS_ARGS})

message(STATUS "OneDNN library: ${ONEDNN_LIB}")
add_definitions(-DPADDLE_WITH_DNNL)
# copy the real so.0 lib to install dir
# it can be directly contained in wheel or capi
if(WIN32)
  set(ONEDNN_SHARED_LIB ${ONEDNN_INSTALL_DIR}/bin/mkldnn.dll)

  file(TO_NATIVE_PATH ${ONEDNN_INSTALL_DIR} NATIVE_ONEDNN_INSTALL_DIR)
  file(TO_NATIVE_PATH ${ONEDNN_SHARED_LIB} NATIVE_ONEDNN_SHARED_LIB)

  add_custom_command(
    OUTPUT ${ONEDNN_LIB}
    COMMAND (copy ${NATIVE_ONEDNN_INSTALL_DIR}\\bin\\dnnl.dll
             ${NATIVE_ONEDNN_SHARED_LIB} /Y)
    COMMAND dumpbin /exports ${ONEDNN_INSTALL_DIR}/bin/mkldnn.dll >
            ${ONEDNN_INSTALL_DIR}/bin/exports.txt
    COMMAND echo LIBRARY mkldnn > ${ONEDNN_INSTALL_DIR}/bin/mkldnn.def
    COMMAND echo EXPORTS >> ${ONEDNN_INSTALL_DIR}/bin/mkldnn.def
    COMMAND
      echo off && (for
                   /f
                   "skip=19 tokens=4"
                   %A
                   in
                   (${ONEDNN_INSTALL_DIR}/bin/exports.txt)
                   do
                   echo
                   %A
                   >>
                   ${ONEDNN_INSTALL_DIR}/bin/mkldnn.def) && echo on
    COMMAND lib /def:${ONEDNN_INSTALL_DIR}/bin/mkldnn.def /out:${ONEDNN_LIB}
            /machine:x64
    COMMENT "Generate mkldnn.lib manually--->"
    DEPENDS ${ONEDNN_PROJECT}
    VERBATIM)
  add_custom_target(onednn_cmd ALL DEPENDS ${ONEDNN_LIB})
else()
  set(ONEDNN_SHARED_LIB ${ONEDNN_INSTALL_DIR}/libdnnl.so.3)
  add_custom_command(
    OUTPUT ${ONEDNN_SHARED_LIB}
    COMMAND ${CMAKE_COMMAND} -E copy ${ONEDNN_LIB} ${ONEDNN_SHARED_LIB}
    DEPENDS ${ONEDNN_PROJECT})
  add_custom_target(onednn_cmd ALL DEPENDS ${ONEDNN_SHARED_LIB})
endif()

# generate a static dummy target to track onednn dependencies
# for cc_library(xxx SRCS xxx.c DEPS onednn)
generate_dummy_static_lib(LIB_NAME "onednn" GENERATOR "onednn.cmake")

target_link_libraries(onednn ${ONEDNN_LIB} ${MKLML_IOMP_LIB})
add_dependencies(onednn ${ONEDNN_PROJECT} onednn_cmd)
