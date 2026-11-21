// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// gin_dynload.h: using-declarations that bring the NCCL host API and the CUDA
// driver API calls of gin_overlap into namespace gin_overlap. The calls go to
// the dlopen wrappers in phi::dynload (phi/backends/dynload/nccl.h,
// cuda_driver.h), so call sites in gin_overlap use the plain API names.
//
// Not covered here:
//   - device-side API used in kernels (ncclGin*, ncclTeamWorld(ncclDevComm
//     const&), ncclGetLsaPointer): __device__ inline functions of
//     nccl_device.h.
//   - the host ncclTeamWorld(ncclComm_t): it shares its name with the device
//     overload, so DECLARE_DYNAMIC_LOAD_NCCL_WRAP cannot wrap it; host code of
//     gin_overlap must not call it.
//   - runtime API such as cudaMemcpyBatchAsync: called directly, not through
//     phi::dynload. On Linux libcudart is linked dynamically (--cudart shared,
//     top-level CMakeLists.txt), so libpaddle.so needs a libcudart that has
//     these entry points (CUDA >= 13.0, checked below).
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/backends/dynload/nccl.h"

// When the CMake gate (collective/CMakeLists.txt) is not met this directory is
// not built; the checks below turn a wrong gate into a compile error.
#if !defined(PADDLE_WITH_NCCL)
#error "gin_overlap needs PADDLE_WITH_NCCL (NCCL on CUDA); check the CMake gate"
#endif
#if NCCL_VERSION_CODE < 23007
#error "gin_overlap needs NCCL >= 2.30.7 for the device (LSA/GIN) API"
#endif
#if CUDART_VERSION < 13000
#error "gin_overlap needs CUDA >= 13.0 for the cudaMemcpyBatchAsync signature"
#endif

namespace gin_overlap {

// NCCL host API: the calls of nccl.h, plus the two host entry points of
// nccl_device.h (ncclDevCommCreate, ncclGetLsaDevicePointer; NCCL >= 2.30.7)
using phi::dynload::ncclCommCount;
using phi::dynload::ncclCommUserRank;
using phi::dynload::ncclCommWindowRegister;
using phi::dynload::ncclDevCommCreate;
using phi::dynload::ncclGetErrorString;
using phi::dynload::ncclGetLsaDevicePointer;
using phi::dynload::ncclGetVersion;
using phi::dynload::ncclMemAlloc;

// CUDA driver API
using phi::dynload::cuDeviceGet;
using phi::dynload::cuDeviceGetAttribute;
using phi::dynload::cuGetErrorString;
using phi::dynload::cuMemAddressReserve;
using phi::dynload::cuMemCreate;
using phi::dynload::cuMemGetAllocationGranularity;
using phi::dynload::cuMemMap;
using phi::dynload::cuMemSetAccess;
// 2D copy: cuda.h defines cuMemcpy2DAsync as a macro that expands to
// cuMemcpy2DAsync_v2. The wrapper must be built under the _v2 name, or dlsym
// returns the v1 entry point, whose parameter struct differs. Call sites keep
// writing cuMemcpy2DAsync(...).
using phi::dynload::cuMemcpy2DAsync_v2;

}  // namespace gin_overlap
