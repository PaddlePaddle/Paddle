# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT LINUX)
  message(STATUS "OpenVINO only support Linux.")
  set(WITH_OPENVINO OFF)
  return()
endif()

include(ExternalProject)

set(OPENVINO_PROJECT "extern_openvino")
set(OPENVINO_PREFIX_DIR ${THIRD_PARTY_PATH}/openvino)
set(OPENVINO_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openvino)
set(OPENVINO_INC_DIR
    "${OPENVINO_INSTALL_DIR}/runtime/include"
    CACHE PATH "OpenVINO include directory." FORCE)
set(SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/openvino)

set(TBB_INC_DIR
    "${OPENVINO_INSTALL_DIR}/runtime/3rdparty/tbb/include"
    CACHE PATH "OpenVINO TBB include directory." FORCE)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)
string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
if(ARCH STREQUAL "x86_64"
   OR ARCH STREQUAL "amd64"
   OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
  set(ARCH intel64)
elseif(ARCH STREQUAL "i386")
  set(ARCH ia32)
endif()
set(LIBDIR "runtime/lib/${ARCH}")
set(TBBDIR "runtime/3rdparty/tbb/lib")

set(OPENVINO_LIB_NAME
    "libopenvino.so.2500"
    CACHE PATH "libopenvino name." FORCE)
set(OPENVINO_PADDLE_LIB_NAME
    "libopenvino_paddle_frontend.so.2500"
    CACHE PATH "libopenvino_paddle_frontend name." FORCE)
set(OPENVINO_CPU_PLUGIN_LIB_NAME
    "libopenvino_intel_cpu_plugin.so"
    CACHE PATH "libopenvino_intel_cpu_plugin name." FORCE)
set(TBB_LIB_NAME
    "libtbb.so.12"
    CACHE PATH "libtbb name." FORCE)

message(STATUS "Set ${OPENVINO_INSTALL_DIR}/${LIBDIR} to runtime path")
message(STATUS "Set ${OPENVINO_INSTALL_DIR}/${TBBDIR} to runtime path")
set(OPENVINO_LIB_DIR ${OPENVINO_INSTALL_DIR}/${LIBDIR})
set(TBB_LIB_DIR ${OPENVINO_INSTALL_DIR}/${TBBDIR})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${OPENVINO_LIB_DIR}"
                        "${TBB_LIB_DIR}")

include_directories(${OPENVINO_INC_DIR}
)# For OpenVINO code to include internal headers.

include_directories(${TBB_INC_DIR}
)# For OpenVINO TBB code to include third_party headers.

if(LINUX)
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/${LIBDIR}/${OPENVINO_LIB_NAME}"
      CACHE FILEPATH "OpenVINO library." FORCE)
  set(OPENVINO_PADDLE_LIB
      "${OPENVINO_INSTALL_DIR}/${LIBDIR}/${OPENVINO_PADDLE_LIB_NAME}"
      CACHE FILEPATH "OpenVINO paddle frontend library." FORCE)
  set(OPENVINO_CPU_PLUGIN_LIB
      "${OPENVINO_INSTALL_DIR}/${LIBDIR}/${OPENVINO_CPU_PLUGIN_LIB_NAME}"
      CACHE FILEPATH "OpenVINO cpu inference library." FORCE)
  set(TBB_LIB
      "${OPENVINO_INSTALL_DIR}/${TBBDIR}/${TBB_LIB_NAME}"
      CACHE FILEPATH "TBB library." FORCE)
else()
  message(ERROR "Only support Linux.")
endif()

if(LINUX)
  set(BUILD_BYPRODUCTS_ARGS ${OPENVINO_LIB} ${TBB_LIB} ${OPENVINO_PADDLE_LIB}
                            ${OPENVINO_CPU_PLUGIN_LIB})
else()
  set(BUILD_BYPRODUCTS_ARGS "")
endif()

set(OPENVINO_COMMIT 07ecdf07d2974410dc1d67d9fa2d3433dcab7865)
file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/openvino/convert.patch
     native_convert)

set(OPENVINO_PATCH_COMMAND
    git checkout -- . && git fetch --depth=1 origin ${OPENVINO_COMMIT} && git
    checkout ${OPENVINO_COMMIT} && patch -Np1 -d ${SOURCE_DIR} <
    ${native_convert} || true)

ExternalProject_Add(
  ${OPENVINO_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${SOURCE_DIR}
  DEPENDS ${OPENVINO_DEPENDS}
  PREFIX ${OPENVINO_PREFIX_DIR}
  UPDATE_COMMAND ""
  #BUILD_ALWAYS        1
  PATCH_COMMAND ${OPENVINO_PATCH_COMMAND}
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_FLAGS=${ONEDNN_CXXFLAG}
             -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
             -DCMAKE_CXX_FLAGS_RELEASE=${ONEDNN_CXXFLAG_RELEASE}
             -DCMAKE_C_FLAGS=${ONEDNN_CFLAG}
             -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${ONEDNN_CFLAG_RELEASE}
             -DCMAKE_INSTALL_PREFIX=${OPENVINO_INSTALL_DIR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DTHREADING=TBB
             -DENABLE_INTEL_CPU=ON
             -DENABLE_INTEL_GPU=OFF
             -DENABLE_INTEL_NPU=OFF
             -DENABLE_HETERO=OFF
             -DENABLE_MULTI=OFF
             -DENABLE_AUTO=OFF
             -DENABLE_TEMPLATE=OFF
             -DENABLE_AUTO_BATCH=OFF
             -DENABLE_PROXY=OFF
             -DENABLE_OV_ONNX_FRONTEND=OFF
             -DENABLE_OV_TF_FRONTEND=OFF
             -DENABLE_OV_TF_LITE_FRONTEND=OFF
             -DENABLE_OV_PYTORCH_FRONTEND=OFF
             -DENABLE_OV_JAX_FRONTEND=OFF
             -DENABLE_OV_IR_FRONTEND=OFF
             -DENABLE_SAMPLES=OFF
             -DENABLE_TESTS=OFF
             -DENABLE_PYTHON=OFF
             -DENABLE_WHEEL=OFF
             -DENABLE_DOCS=OFF
             -DENABLE_CPPLINT=OFF
             -DENABLE_CLANG_FORMAT=OFF
             -DENABLE_NCC_STYLE=OFF
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${OPENVINO_INSTALL_DIR}
  BUILD_BYPRODUCTS ${BUILD_BYPRODUCTS_ARGS})

message(STATUS "OpenVINO library: ${OPENVINO_LIB}")
message(STATUS "OpenVINO Paddle library: ${OPENVINO_PADDLE_LIB}")
message(STATUS "OpenVINO CPU Inference library: ${OPENVINO_CPU_PLUGIN_LIB}")
message(STATUS "OpenVINO TBB library: ${TBB_LIB}")
add_definitions(-DPADDLE_WITH_OPENVINO)

add_library(openvino SHARED IMPORTED GLOBAL)
add_library(openvino_paddle SHARED IMPORTED GLOBAL)
add_library(openvino_cpu_plugin SHARED IMPORTED GLOBAL)
add_library(tbb SHARED IMPORTED GLOBAL)
set_property(TARGET openvino PROPERTY IMPORTED_LOCATION ${OPENVINO_LIB})
set_property(TARGET openvino_paddle PROPERTY IMPORTED_LOCATION
                                             ${OPENVINO_PADDLE_LIB})
set_property(TARGET openvino_cpu_plugin PROPERTY IMPORTED_LOCATION
                                                 ${OPENVINO_CPU_PLUGIN_LIB})
set_property(TARGET tbb PROPERTY IMPORTED_LOCATION ${TBB_LIB})
add_dependencies(openvino ${OPENVINO_PROJECT})
add_dependencies(openvino_paddle ${OPENVINO_PROJECT})
add_dependencies(openvino_cpu_plugin ${OPENVINO_PROJECT})
add_dependencies(tbb ${OPENVINO_PROJECT})
