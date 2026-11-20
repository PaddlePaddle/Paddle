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

// gin_dynload.h —— 这一层只做一件事：把 gin_overlap 用到的 NCCL host API 和
// CUDA 驱动 API 的名字引进 namespace gin_overlap。
//
// 为什么需要：Paddle 既不链 libnccl 也不链 libcuda，两套 API 都走 phi::dynload
//   的 dlopen 包装（paddle/phi/backends/dynload/{nccl,cuda_driver}.h，
//   phi::dynload::ncclMemAlloc 这类是 struct 对象，operator() 里 dlsym 之后再
//   调）。gin_overlap 的代码是按裸名字写的，这里用 using 声明把名字引进本命名
//   空间：命名空间内的非限定查找先命中这些 using 声明就停下，所以各处调用点
//   一行都不用改，实际走的是 dlopen 的那一份。
//
// 不包在这里的：
//   * kernel 里用的设备侧 API（ncclGin* / ncclTeamWorld(ncclDevComm const&) /
//     ncclGetLsaPointer）是 nccl_device.h 里的 __device__ inline，直接编进
//     cubin，没有动态符号可包。
//   * nccl_device.h 里那个 host 版 ncclTeamWorld(ncclComm_t) 故意不包：它和
//     设备版同名重载，DECLARE_DYNAMIC_LOAD_NCCL_WRAP 里的 decltype(&::__name)
//     对重载名无法定型；host 侧也没有一处用它（put.cu / signal.cu 用的都是收
//     ncclDevComm 的设备版）。
//   * cudaMemcpyBatchAsync 等 runtime API 不用包：libcudart 是静态链进来的。
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/backends/dynload/nccl.h"

// 不支持的配置继续报错（不静默退化）：CMake 那边的门控（WITH_NCCL + NCCL
// 版本 + nccl_device.h 存在 + arch 含 90/100）本来就不该把这个目录编进来，
// 这两条是防门控写错的兜底。
#if !defined(PADDLE_WITH_NCCL)
#error "gin_overlap needs PADDLE_WITH_NCCL (NCCL on CUDA); check the CMake gate"
#endif
#if NCCL_VERSION_CODE < 23007
#error "gin_overlap needs NCCL >= 2.30.7 for the device (LSA/GIN) API"
#endif

namespace gin_overlap {

// NCCL host API（nccl.h 的 + nccl_device.h 里那两个 host 入口，后两个在
// phi/backends/dynload/nccl.h 的 NCCL >= 2.30.7 那一组里）
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
// 2D 拷贝：cuda.h 里 cuMemcpy2DAsync 是个宏，展开成带版本后缀的
// cuMemcpy2DAsync_v2（cuda.h:120 `#define cuMemcpy2DAsync
// __CUDA_API_PTSZ(cuMemcpy2DAsync_v2)`），libcuda 导出的也是 _v2 那个名字
// （v1 的参数是老的 CUDA_MEMCPY2D_v1 结构）。包装器必须按 _v2 这个真名建，
// 否则 dlsym("cuMemcpy2DAsync") 会拿到 v1、再按 v2 的签名去调。调用点写的
// cuMemcpy2DAsync(...) 由 cuda.h 的宏改写成 _v2，所以不用动（Python 侧原来
// 用 ctypes 调的也正是 libcuda.so.1 的 cuMemcpy2DAsync_v2）。
using phi::dynload::cuMemcpy2DAsync_v2;

}  // namespace gin_overlap
