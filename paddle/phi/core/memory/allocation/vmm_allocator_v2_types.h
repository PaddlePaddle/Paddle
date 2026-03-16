// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cstdint>
#include <memory>

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cuda_driver.h"
using VmmDevicePtr = CUdeviceptr;
using VmmAllocHandle = CUmemGenericAllocationHandle;
#else
using VmmDevicePtr = uintptr_t;
using VmmAllocHandle = uint64_t;
#endif

namespace paddle {
namespace memory {
namespace allocation {

// V2 keeps the bottom-layer shared types independent from the best-fit layer
// so that CUDAVirtualMemAllocatorV2 can be reviewed and compiled separately.
enum class PoolType : uint8_t {
  kStable = 0,
  kLongLived = 1,
  kTransient = 2,
  kOversized = 3,
};

// Fixed-size handle metadata shared by BlockPartV2. This is the bottom-layer
// object that later remap / IPC / shared-handle lifetime management will build
// on top of.
struct VmmHandleMeta {
  VmmDevicePtr base;
  size_t size;
  VmmAllocHandle handle;
  int device;
};

// A logical slice of one fixed-size VMM handle. Higher layers may split one
// allocation into multiple BlockPartV2 entries and later reuse the same
// representation for remap / GAP / IPC bookkeeping.
struct BlockPartV2 {
  std::shared_ptr<VmmHandleMeta> chunk;
  size_t chunk_rel_off;
  size_t len;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
