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

// gin_dynload.h —— 用 using 声明把 gin_overlap 用到的 NCCL host API 和 CUDA
// 驱动 API 引进 namespace gin_overlap，实际调用的是 phi::dynload 里的 dlopen
// 包装（phi/backends/dynload/nccl.h、cuda_driver.h）。gin_overlap 里的调用点
// 按原名书写即可。
//
// 不经过这里的：
//   - kernel 里用的设备侧 API（ncclGin*、ncclTeamWorld(ncclDevComm const&)、
//     ncclGetLsaPointer）：nccl_device.h 里的 __device__ inline 函数。
//   - host 版 ncclTeamWorld(ncclComm_t)：和设备版重载同名，不能用
//     DECLARE_DYNAMIC_LOAD_NCCL_WRAP 包装；gin_overlap 的 host 代码不能调用它。
//   - cudaMemcpyBatchAsync 等 runtime API：libcudart 是静态链接的。
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/backends/dynload/nccl.h"

// CMake 门控（collective/CMakeLists.txt）不满足时这个目录不会参与编译；下面两条
// 在门控写错时直接报编译错误。
#if !defined(PADDLE_WITH_NCCL)
#error "gin_overlap needs PADDLE_WITH_NCCL (NCCL on CUDA); check the CMake gate"
#endif
#if NCCL_VERSION_CODE < 23007
#error "gin_overlap needs NCCL >= 2.30.7 for the device (LSA/GIN) API"
#endif

namespace gin_overlap {

// NCCL host API：nccl.h 里的，加上 nccl_device.h 的两个 host 入口
// （ncclDevCommCreate、ncclGetLsaDevicePointer，要求 NCCL >= 2.30.7）
using phi::dynload::ncclCommCount;
using phi::dynload::ncclCommUserRank;
using phi::dynload::ncclCommWindowRegister;
using phi::dynload::ncclDevCommCreate;
using phi::dynload::ncclGetErrorString;
using phi::dynload::ncclGetLsaDevicePointer;
using phi::dynload::ncclGetVersion;
using phi::dynload::ncclMemAlloc;

// CUDA 驱动 API
using phi::dynload::cuDeviceGet;
using phi::dynload::cuDeviceGetAttribute;
using phi::dynload::cuGetErrorString;
using phi::dynload::cuMemAddressReserve;
using phi::dynload::cuMemCreate;
using phi::dynload::cuMemGetAllocationGranularity;
using phi::dynload::cuMemMap;
using phi::dynload::cuMemSetAccess;
// 2D 拷贝：cuda.h 把 cuMemcpy2DAsync 定义成宏，展开为 cuMemcpy2DAsync_v2。
// 包装器必须按 _v2 这个名字建，否则 dlsym 拿到的是参数结构不同的 v1。调用点
// 照常写 cuMemcpy2DAsync(...)。
using phi::dynload::cuMemcpy2DAsync_v2;

}  // namespace gin_overlap
