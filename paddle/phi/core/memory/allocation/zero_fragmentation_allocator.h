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
#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <utility>

#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

namespace paddle {
namespace memory {
namespace allocation {

class ZeroFragmentationAllocator : public AutoGrowthBestFitAllocator {
 public:
  ZeroFragmentationAllocator(std::shared_ptr<Allocator> underlying_allocator,
                             size_t alignment,
                             size_t chunk_size = 0,
                             bool allow_free_idle_chunk = true,
                             int extra_padding_size = 0)
      : AutoGrowthBestFitAllocator(underlying_allocator,
                                   alignment,
                                   chunk_size,
                                   allow_free_idle_chunk,
                                   extra_padding_size) {}

 protected:
  phi::Allocation* AllocateImpl(size_t unaligned_size) override;

  void FreeImpl(phi::Allocation* allocation) override;

  uint64_t ReleaseImpl(const phi::Place& place) override;

 private:
  void HoldZeroFragmentationBlocks(phi::Allocation* allocation);
  void FreeZeroFragmentationBlocks();

 private:
  BlockIt zero_fragmentation_block_{};
  const BlockIt nulliter{};
};

class ZeroFragmentationAllocatorManager {
 public:
  static ZeroFragmentationAllocatorManager& Instance() noexcept {
    static ZeroFragmentationAllocatorManager instance;
    return instance;
  }

  ZeroFragmentationAllocatorManager(const ZeroFragmentationAllocatorManager&) =
      delete;
  ZeroFragmentationAllocatorManager& operator=(
      const ZeroFragmentationAllocatorManager&) = delete;
  ZeroFragmentationAllocatorManager(ZeroFragmentationAllocatorManager&&) =
      delete;
  ZeroFragmentationAllocatorManager& operator=(
      ZeroFragmentationAllocatorManager&&) = delete;

  // Enable/disable functionality
  void Enable() {
    std::lock_guard<SpinLock> guard(spinlock_);
    enabled_ = true;
  }
  void Disable() {
    std::lock_guard<SpinLock> guard(spinlock_);
    enabled_ = false;
  }
  bool IsEnabled() {
    std::lock_guard<SpinLock> guard(spinlock_);
    return enabled_;
  }

  // Zero fragmentation state management
  void EnterZeroFragmentationMode() {
    std::lock_guard<SpinLock> guard(spinlock_);
    zero_frag_mode_ = true;
  }
  void ExitZeroFragmentationMode() {
    std::lock_guard<SpinLock> guard(spinlock_);
    zero_frag_mode_ = false;
  }
  bool IsZeroFragmentationMode() {
    std::lock_guard<SpinLock> guard(spinlock_);
    return enabled_ && zero_frag_mode_;
  }
  void BeginPreallocation() {
    std::lock_guard<SpinLock> guard(spinlock_);
    preallocating_ = true;
  }
  void EndPreallocation() {
    std::lock_guard<SpinLock> guard(spinlock_);
    preallocating_ = false;
  }
  bool IsPreallocating() {
    std::lock_guard<SpinLock> guard(spinlock_);
    return enabled_ && zero_frag_mode_ && preallocating_;
  }

 private:
  ZeroFragmentationAllocatorManager() = default;

  SpinLock spinlock_;

  bool enabled_{false};
  bool zero_frag_mode_{false};
  bool preallocating_{false};
};

class ZeroFragmentationAllocatorGuard {
 public:
  ZeroFragmentationAllocatorGuard() {
    ZeroFragmentationAllocatorManager::Instance().EnterZeroFragmentationMode();
  }

  ~ZeroFragmentationAllocatorGuard() {
    ZeroFragmentationAllocatorManager::Instance().ExitZeroFragmentationMode();
  }

  ZeroFragmentationAllocatorGuard(const ZeroFragmentationAllocatorGuard&) =
      delete;
  ZeroFragmentationAllocatorGuard& operator=(
      const ZeroFragmentationAllocatorGuard&) = delete;
  ZeroFragmentationAllocatorGuard(ZeroFragmentationAllocatorGuard&& other) =
      delete;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
