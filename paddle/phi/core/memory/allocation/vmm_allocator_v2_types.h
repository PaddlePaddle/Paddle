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

#include <cstdint>
#include <memory>
#include <vector>

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cuda_driver.h"
#endif
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/core/platform/device/gpu/gpu_types.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

#if defined(PADDLE_WITH_CUDA)
using VMMDevicePtr = CUdeviceptr;
using VMMAllocHandle = CUmemGenericAllocationHandle;
#else
using VMMDevicePtr = uintptr_t;
using VMMAllocHandle = uint64_t;
#endif

// V2 keeps the bottom-layer shared types independent from the best-fit layer
// so that CUDAVirtualMemAllocatorV2 can be reviewed and compiled separately.
enum class PoolType : uint8_t {
  kSmall = 0,
  kLarge = 1,
};

// Fixed-size handle metadata returned by the bottom VMM provider. Upper layers
// may later reference these handles from block-level views.
struct VMMHandleMeta {
  VMMHandleMeta() = default;

  VMMHandleMeta(VMMDevicePtr base,
                size_t size,
                VMMAllocHandle handle,
                int device)
      : base_(base), size_(size), handle_(handle), device_(device) {}

  VMMDevicePtr base() const { return base_; }
  size_t size() const { return size_; }
  VMMAllocHandle handle() const { return handle_; }
  int device() const { return device_; }

 private:
  VMMDevicePtr base_{0};
  size_t size_{0};
  VMMAllocHandle handle_{0};
  int device_{0};
};

// HandleLayout is a lightweight allocation-level handle list returned by the
// bottom VMM provider. It is used to bootstrap upper-layer block state.
using HandleLayout = std::vector<std::shared_ptr<VMMHandleMeta>>;

enum class BlockType : uint8_t {
  kActive = 0,
  kFree = 1,
  kUnmappedFree = 2,
};

struct BlockV2 {
  static BlockV2 MakeMappedBlock(BlockType type,
                                 void* ptr,
                                 size_t size,
                                 PoolType pool_type) {
    BlockV2 block;
    block.Reset(ptr, size, type, pool_type);
    return block;
  }

  static BlockV2 MakeUnmappedFreeBlock(void* ptr,
                                       size_t size,
                                       PoolType pool_type) {
    BlockV2 block;
    block.Reset(ptr, size, BlockType::kUnmappedFree, pool_type);
    return block;
  }

  bool IsActive() const { return type_ == BlockType::kActive; }
  bool IsFree() const { return type_ == BlockType::kFree; }
  bool IsMappedFree() const { return IsFree(); }
  bool IsUnmappedFree() const { return type_ == BlockType::kUnmappedFree; }
  void* ptr() const { return ptr_; }
  size_t size() const { return size_; }
  uint8_t* begin_ptr() const { return reinterpret_cast<uint8_t*>(ptr_); }
  uint8_t* end_ptr() const { return begin_ptr() + size_; }
  VMMDevicePtr begin_va() const {
    return reinterpret_cast<VMMDevicePtr>(begin_ptr());
  }
  VMMDevicePtr end_va() const { return begin_va() + size_; }
  bool IsAdjacentBefore(const BlockV2& next) const {
    return end_ptr() == next.begin_ptr();
  }
  bool CanMergeAdjacentFreeBlock(const BlockV2& next) const {
    return IsFree() && next.IsFree() && IsAdjacentBefore(next);
  }
  bool CanMergeAdjacentUnmappedFreeBlock(const BlockV2& next) const {
    return IsUnmappedFree() && next.IsUnmappedFree() && IsAdjacentBefore(next);
  }
  BlockV2 MakeMappedFreeSubBlock(size_t offset, size_t len) const {
    return MakeMappedBlock(
        BlockType::kFree, begin_ptr() + offset, len, pool_type_);
  }
  BlockV2 MakeMappedActiveSubBlock(size_t offset, size_t len) const {
    return MakeMappedBlock(
        BlockType::kActive, begin_ptr() + offset, len, pool_type_);
  }
  BlockV2 MakeUnmappedFreeSubBlock(size_t offset, size_t len) const {
    return MakeUnmappedFreeBlock(begin_ptr() + offset, len, pool_type_);
  }
  void MarkActive() { type_ = BlockType::kActive; }
  void MarkFree() { type_ = BlockType::kFree; }
  void Reset(void* ptr, size_t size, BlockType type, PoolType pool_type) {
    ptr_ = ptr;
    size_ = size;
    type_ = type;
    pool_type_ = pool_type;
  }
  void TrimToPrefix(size_t keep) { size_ = keep; }
  void TrimToSuffix(size_t trim, size_t keep) {
    ptr_ = reinterpret_cast<uint8_t*>(ptr_) + trim;
    size_ = keep;
  }
  void MergeAdjacentBlock(const BlockV2& src) { size_ += src.size_; }
  void MergeAdjacentUnmappedFreeBlock(const BlockV2& src) {
    size_ += src.size_;
  }

  void* ptr_{nullptr};
  size_t size_{0};
  BlockType type_{BlockType::kUnmappedFree};

  PoolType pool_type_{PoolType::kLarge};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
