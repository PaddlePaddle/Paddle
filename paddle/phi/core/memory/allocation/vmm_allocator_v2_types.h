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

#pragma once

#include <memory>
#include <vector>

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

// Fixed-size handle metadata returned by the bottom VMM provider. Upper layers
// may later reference these handles from block-level views, remap metadata, or
// IPC export state.
struct VmmHandleMeta {
  VmmDevicePtr base;
  size_t size;
  VmmAllocHandle handle;
  int device;
};

// HandleLayout is a lightweight allocation-level handle list returned by the
// bottom VMM provider. It is only used to bootstrap upper-layer block state or
// answer allocation-level IPC/export queries.
using HandleLayout = std::vector<std::shared_ptr<VmmHandleMeta>>;

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
