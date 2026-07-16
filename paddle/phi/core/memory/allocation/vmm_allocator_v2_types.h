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

#if defined(PADDLE_WITH_CUDA)
struct CUDAEventGuard {
  gpuEvent_t event{nullptr};

  explicit CUDAEventGuard(gpuEvent_t e) : event(e) {}
  ~CUDAEventGuard() {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }

  CUDAEventGuard(const CUDAEventGuard&) = delete;
  CUDAEventGuard& operator=(const CUDAEventGuard&) = delete;
};

class VMMRemapEventAllocation {
 public:
  virtual ~VMMRemapEventAllocation() = default;
  virtual bool SetVMMRemapEvent(gpuStream_t stream,
                                std::shared_ptr<CUDAEventGuard> event) = 0;
};

struct VMMBlockRemapState {
  gpuStream_t stream{nullptr};
  std::shared_ptr<CUDAEventGuard> event;
};
#endif

enum class PoolType : uint8_t {
  kSmall = 0,
  kLarge = 1,
};

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

  bool IsOwnedByRemapDestination() const { return owned_by_remap_destination_; }
  void MarkOwnedByRemapDestination() { owned_by_remap_destination_ = true; }
  void RestoreOriginalOwnership() { owned_by_remap_destination_ = false; }

 private:
  VMMDevicePtr base_{0};
  size_t size_{0};
  VMMAllocHandle handle_{0};
  int device_{0};
  bool owned_by_remap_destination_{false};
};

using HandleLayout = std::vector<std::shared_ptr<VMMHandleMeta>>;

struct IPCPartDescriptor {
  VMMDevicePtr handle_base;
  size_t handle_size;
  VMMAllocHandle handle;
  int device;
  size_t handle_rel_off;
  size_t len;
};

enum class BlockType : uint8_t {
  kActive = 0,
  kFree = 1,
  kUnmappedFree = 2,
};

enum class BlockRestoreMappedFreeResult : uint8_t {
  kOutside = 0,
  kRangeExceedsBlock = 1,
  kBuilt = 2,
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
  std::pair<VMMDevicePtr, size_t> va_range() const {
    return {begin_va(), size_};
  }
  bool ContainsVARange(VMMDevicePtr va, size_t size) const {
    return size > 0 && va >= begin_va() && va < end_va() &&
           size <= end_va() - va;
  }
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
    auto block = MakeMappedBlock(
        BlockType::kFree, begin_ptr() + offset, len, pool_type_);
#if defined(PADDLE_WITH_CUDA)
    block.CopyRemapSafetyFrom(*this);
#endif
    return block;
  }
  BlockV2 MakeMappedActiveSubBlock(size_t offset, size_t len) const {
    auto block = MakeMappedBlock(
        BlockType::kActive, begin_ptr() + offset, len, pool_type_);
#if defined(PADDLE_WITH_CUDA)
    block.ClearRemapSafety();
#endif
    return block;
  }
  BlockV2 MakeUnmappedFreeSubBlock(size_t offset, size_t len) const {
    return MakeUnmappedFreeBlock(begin_ptr() + offset, len, pool_type_);
  }
  BlockRestoreMappedFreeResult BuildRestoreMappedFreeSegments(
      VMMDevicePtr va, size_t size, std::vector<BlockV2>* segments) const {
    if (!IsUnmappedFree() || va < begin_va() || va >= end_va()) {
      return BlockRestoreMappedFreeResult::kOutside;
    }
    if (size > end_va() - va) {
      return BlockRestoreMappedFreeResult::kRangeExceedsBlock;
    }

    segments->clear();
    const size_t prefix = va - begin_va();
    const size_t suffix = end_va() - (va + size);
    if (prefix > 0) {
      segments->push_back(MakeUnmappedFreeSubBlock(0, prefix));
    }
    segments->push_back(MakeMappedBlock(
        BlockType::kFree, reinterpret_cast<void*>(va), size, pool_type_));
    if (suffix > 0) {
      segments->push_back(MakeUnmappedFreeSubBlock(prefix + size, suffix));
    }
    return BlockRestoreMappedFreeResult::kBuilt;
  }
  void MarkActive() {
    type_ = BlockType::kActive;
#if defined(PADDLE_WITH_CUDA)
    ClearRemapSafety();
#endif
  }
  void MarkFree() { type_ = BlockType::kFree; }
  void MarkUnmappedFree() { type_ = BlockType::kUnmappedFree; }
  void Reset(void* ptr, size_t size, BlockType type, PoolType pool_type) {
    ptr_ = ptr;
    size_ = size;
    type_ = type;
    pool_type_ = pool_type;
#if defined(PADDLE_WITH_CUDA)
    ClearRemapSafety();
#endif
  }
  void TrimToPrefix(size_t keep) { size_ = keep; }
  void TrimToSuffix(size_t trim, size_t keep) {
    ptr_ = reinterpret_cast<uint8_t*>(ptr_) + trim;
    size_ = keep;
  }
  void MergeAdjacentBlock(const BlockV2& src) {
    size_ += src.size_;
#if defined(PADDLE_WITH_CUDA)
    AppendRemapSafetyFrom(src);
#endif
  }
  void MergeAdjacentUnmappedFreeBlock(const BlockV2& src) {
    size_ += src.size_;
  }

  void* ptr_{nullptr};
  size_t size_{0};
  BlockType type_{BlockType::kUnmappedFree};
  PoolType pool_type_{PoolType::kLarge};

#if defined(PADDLE_WITH_CUDA)
  void ClearRemapSafety() {
    owning_stream_ = nullptr;
    remap_safe_event_.reset();
    remap_pending_states_.clear();
    remap_safety_unknown_ = false;
  }
  void SetRemapSafety(gpuStream_t stream,
                      std::shared_ptr<CUDAEventGuard> event) {
    ClearRemapSafety();
    if (stream == nullptr && event == nullptr) {
      remap_safety_unknown_ = true;
      return;
    }
    owning_stream_ = stream;
    remap_safe_event_ = std::move(event);
  }
  void CopyRemapSafetyFrom(const BlockV2& src) {
    owning_stream_ = src.owning_stream_;
    remap_safe_event_ = src.remap_safe_event_;
    remap_pending_states_ = src.remap_pending_states_;
    remap_safety_unknown_ = src.remap_safety_unknown_;
  }
  void AppendRemapSafety(gpuStream_t stream,
                         std::shared_ptr<CUDAEventGuard> event) {
    if (stream == nullptr && event == nullptr) {
      return;
    }
    if (owning_stream_ == stream && remap_safe_event_.get() == event.get()) {
      return;
    }
    for (const auto& state : remap_pending_states_) {
      if (state.stream == stream && state.event.get() == event.get()) {
        return;
      }
    }
    if (owning_stream_ == nullptr && remap_safe_event_ == nullptr) {
      owning_stream_ = stream;
      remap_safe_event_ = std::move(event);
      return;
    }
    remap_pending_states_.push_back({stream, std::move(event)});
  }
  void AppendRemapSafetyFrom(const BlockV2& src) {
    remap_safety_unknown_ = remap_safety_unknown_ || src.remap_safety_unknown_;
    AppendRemapSafety(src.owning_stream_, src.remap_safe_event_);
    for (const auto& state : src.remap_pending_states_) {
      AppendRemapSafety(state.stream, state.event);
    }
  }
  bool HasUnknownRemapSafety() const { return remap_safety_unknown_; }

  gpuStream_t owning_stream_{nullptr};
  std::shared_ptr<CUDAEventGuard> remap_safe_event_;
  std::vector<VMMBlockRemapState> remap_pending_states_;
  bool remap_safety_unknown_{false};
#endif
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
