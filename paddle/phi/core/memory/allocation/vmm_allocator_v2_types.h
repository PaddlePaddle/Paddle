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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
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

struct ChunkV2;

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

// HandleLayout is a lightweight allocation-level handle list returned by the
// bottom VMM provider. It is only used to bootstrap block-level BlockPartV2
// state or answer allocation-level IPC/export queries.
using HandleLayout = std::vector<std::shared_ptr<VmmHandleMeta>>;

// A logical slice of one fixed-size VMM handle. This is the block-level view
// owned by VMMAutoGrowthBestFitAllocatorV2 and is updated by split / merge /
// remap after the initial HandleLayout has been consumed. Future IPC export
// still exports whole handles at the driver layer; BlockPartV2 carries the
// slice metadata needed to rebuild the logical tensor view on import.
struct BlockPartV2 {
  std::shared_ptr<VmmHandleMeta> chunk;
  size_t chunk_rel_off;
  size_t len;
};

enum class BlockType : uint8_t {
  kActive = 0,
  kFree = 1,
  kGap = 2,
};

struct PerPoolStats {
  std::atomic<size_t> alloc_count{0};
  std::atomic<size_t> alloc_bytes{0};
  std::atomic<size_t> free_count{0};
  std::atomic<size_t> remap_bytes{0};
  std::atomic<size_t> offload_candidate_bytes{0};

  std::atomic<size_t> live_bytes{0};
  std::atomic<size_t> free_bytes{0};
  std::atomic<size_t> gap_bytes{0};
};

struct BlockV2 {
  void* ptr_{nullptr};
  size_t size_{0};
  BlockType type_{BlockType::kGap};
  ChunkV2* chunk_{nullptr};
  std::vector<BlockPartV2> parts_;
  PoolType pool_type_{PoolType::kTransient};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  gpuStream_t owning_stream_{nullptr};
  gpuStream_t last_use_stream_{nullptr};
  gpuEvent_t remap_safe_event_{nullptr};
#endif
  bool ipc_exported_{false};
};

using BlockList = std::list<BlockV2>;
using BlockListIt = BlockList::iterator;
using PtrBlockMap = std::unordered_map<void*, BlockListIt>;

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
