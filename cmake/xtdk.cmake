# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cmake/xtdk.cmake
# ----------------
# Locate the XTDK (XPU Tensor Developer Kit) and XRE (XPU Runtime Environment)
# required to build the CINN XPU (M100/Houyi) backend.
#
# Mirrors the pattern used by cmake/hip.cmake for the HygonDCU/ROCm backend:
#
#   cmake/hip.cmake   -> cmake/xtdk.cmake
#   ROCM_PATH         -> XTDK_PATH  (compiler toolchain)
#   ROCM_PATH         -> XRE_PATH   (runtime library, separate package)
#   ROCM_HIPRTC_LIB   -> XTDK_XPUJITC_LIB + XRE_XCUDA_LIB + XRE_CUDART_LIB
#
# Usage (cmake configure):
#
#   cmake <src> -DWITH_XPU=ON \
#               -DXTDK_PATH=/opt/xtdk-llvm19-ubuntu2004_x86_64 \
#               -DXRE_PATH=/opt/xre
#
# Both paths can alternatively be provided via environment variables
# XTDK_PATH and XRE_PATH.

if(NOT WITH_XPU)
  return()
endif()

# ---------------------------------------------------------------------------
# 1. XTDK_PATH: compiler toolchain + xpurtc JIT library (libxpujitc.so)
# ---------------------------------------------------------------------------
if(NOT DEFINED XTDK_PATH)
  if(DEFINED ENV{XTDK_PATH})
    set(XTDK_PATH
        $ENV{XTDK_PATH}
        CACHE PATH "Path to the XTDK installation directory")
  else()
    message(
      FATAL_ERROR
        "XTDK_PATH is not set.\n"
        "Provide it via -DXTDK_PATH=<dir> or export XTDK_PATH=<dir>.\n"
        "Example: /opt/xtdk-llvm19-ubuntu2004_x86_64")
  endif()
endif()
message(STATUS "XTDK_PATH: ${XTDK_PATH}")

# Clang 19 kernel include root (lib/clang/19/include/)
set(XTDK_CLANG_INCLUDE
    "${XTDK_PATH}/lib/clang/19/include"
    CACHE PATH "XTDK Clang 19 kernel include directory")
if(NOT EXISTS "${XTDK_CLANG_INCLUDE}")
  message(
    FATAL_ERROR
      "XTDK Clang include directory not found: ${XTDK_CLANG_INCLUDE}\n"
      "Check that XTDK_PATH points to the correct XTDK installation.")
endif()
message(STATUS "XTDK Clang include: ${XTDK_CLANG_INCLUDE}")

# xpurtc JIT compiler shared library.
#
# NOTE: We deliberately link the SONAME-isolated copy shlib/libxpujitc_xtdk.so
# instead of the stock shlib/libxpujitc.so. Rationale: the XPU third-party (XHPC)
# package ships its OWN libxpujitc.so with an ABI-incompatible xpurtc API that
# libxpu_blas.so depends on. Both files share SONAME "libxpujitc.so", so a linker
# can only pick one and neither is a superset. To let CINN use the XTDK jitc while
# libxpu_blas keeps using the XHPC jitc, the XTDK copy is given a distinct SONAME
# (libxpujitc_xtdk.so) and its colliding xpurtc::CompileContext destructor symbol
# is renamed so it no longer interposes the XHPC one at runtime. See build notes.
find_library(
  XTDK_XPUJITC_LIB
  NAMES xpujitc_xtdk
  PATHS "${XTDK_PATH}/shlib"
  NO_DEFAULT_PATH)
if(NOT XTDK_XPUJITC_LIB)
  message(
    FATAL_ERROR
      "Cannot find libxpujitc_xtdk.so under ${XTDK_PATH}/shlib.\n"
      "This is the SONAME-isolated copy of libxpujitc.so required to avoid a "
      "clash with the XHPC third-party libxpujitc.so. Create it with:\n"
      "  cp libxpujitc.so libxpujitc_xtdk.so && \\\n"
      "  patchelf --set-soname libxpujitc_xtdk.so libxpujitc_xtdk.so\n"
      "and rename its xpurtc::CompileContext destructor symbol as documented.")
endif()
message(STATUS "XTDK libxpujitc (isolated): ${XTDK_XPUJITC_LIB}")

# XTDK top-level include (xpurtc.h, xpu_compile_module.h)
# NOTE: Added as a SYSTEM include so it is searched AFTER CINN's bundled
# LLVM 13.0.1 headers (added via a normal -I in cmake/cinn.cmake). XTDK ships a
# full LLVM 19 tree under include/llvm that would otherwise shadow CINN's
# LLVM 13.0.1 and break the CINN host codegen build. XPU backend files only need
# the <xpu/...> headers from XTDK, which are unique to this dir and still
# resolve correctly as a system include.
if(EXISTS "${XTDK_PATH}/include")
  include_directories(SYSTEM "${XTDK_PATH}/include")
endif()

# ---------------------------------------------------------------------------
# 2. XRE_PATH: XPU Runtime Environment (xcuda / cudart CUDA-compat runtime)
#    Provides: libxpucuda.so, libcudart.so, cuda_runtime_api.h, cuda_fp16.h,
#              cuda_bf16.h, cooperative_groups.h
# ---------------------------------------------------------------------------
if(NOT DEFINED XRE_PATH)
  if(DEFINED ENV{XRE_PATH})
    set(XRE_PATH
        $ENV{XRE_PATH}
        CACHE PATH "Path to the XRE (XPU Runtime Environment) installation")
  else()
    message(
      FATAL_ERROR
        "XRE_PATH is not set.\n"
        "Provide it via -DXRE_PATH=<dir> or export XRE_PATH=<dir>.\n"
        "The XRE provides libxpucuda.so and libcudart.so for M100.")
  endif()
endif()
message(STATUS "XRE_PATH: ${XRE_PATH}")

set(XRE_INCLUDE_DIR
    "${XRE_PATH}/include"
    CACHE PATH "XRE runtime include directory")
set(XRE_LIB_DIR
    "${XRE_PATH}/so"
    CACHE PATH "XRE runtime library directory")

if(NOT EXISTS "${XRE_INCLUDE_DIR}")
  message(
    FATAL_ERROR "XRE include directory not found: ${XRE_INCLUDE_DIR}\n"
                "Check that XRE_PATH points to the correct XRE installation.")
endif()
include_directories(${XRE_INCLUDE_DIR})
message(STATUS "XRE include: ${XRE_INCLUDE_DIR}")

find_library(
  XRE_XCUDA_LIB
  NAMES xpucuda
  PATHS "${XRE_LIB_DIR}"
  NO_DEFAULT_PATH)
if(NOT XRE_XCUDA_LIB)
  message(FATAL_ERROR "Cannot find libxpucuda.so under ${XRE_LIB_DIR}.\n"
                      "Check that XRE_PATH is set correctly.")
endif()
message(STATUS "XRE libxpucuda: ${XRE_XCUDA_LIB}")

find_library(
  XRE_CUDART_LIB
  NAMES cudart
  PATHS "${XRE_LIB_DIR}"
  NO_DEFAULT_PATH)
if(NOT XRE_CUDART_LIB)
  message(FATAL_ERROR "Cannot find libcudart.so under ${XRE_LIB_DIR}.\n"
                      "Check that XRE_PATH is set correctly.")
endif()
message(STATUS "XRE libcudart: ${XRE_CUDART_LIB}")

# Aggregate into a single variable consumed by cinn.cmake and generic.cmake,
# mirroring ROCM_HIPRTC_LIB which bundles the one key HIP runtime lib.
set(XPU_XTDK_LIBS
    ${XTDK_XPUJITC_LIB} ${XRE_XCUDA_LIB} ${XRE_CUDART_LIB}
    CACHE INTERNAL "XPU M100 XTDK + XRE link libraries")
message(STATUS "XPU_XTDK_LIBS: ${XPU_XTDK_LIBS}")

# Export LD_LIBRARY_PATH hint for runtime (libxpujitc.so lives in shlib/)
message(STATUS "Ensure LD_LIBRARY_PATH includes:\n"
               "  ${XTDK_PATH}/shlib  (libxpujitc.so)\n"
               "  ${XRE_LIB_DIR}  (libxpucuda.so, libcudart.so)")
