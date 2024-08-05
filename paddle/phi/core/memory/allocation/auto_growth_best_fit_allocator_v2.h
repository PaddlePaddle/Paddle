// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
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

class AutoGrowthBestFitAllocatorV2 : public AutoGrowthBestFitAllocator {
 public:
  AutoGrowthBestFitAllocatorV2(
      const std::shared_ptr<Allocator> &underlying_allocator,
      size_t alignment,
      phi::GPUPlace place,
      size_t chunk_size = 0,
      bool allow_free_idle_chunk = true,
      int extra_padding_size = 0);

 protected:
  phi::Allocation *AllocateImpl(size_t size) override;

 private:
  phi::GPUPlace place_;
  bool is_first_switch_to_regular_{true};
};

class AutoGrowthBestFitAllocatorV2State {
 public:
  AutoGrowthBestFitAllocatorV2State() = default;

  ~AutoGrowthBestFitAllocatorV2State() {}

  void SetWarmup(bool warmup) { is_warmup_ = warmup; }

  bool IsWarmup() { return is_warmup_; }

  static AutoGrowthBestFitAllocatorV2State &GetInstance() {
    static AutoGrowthBestFitAllocatorV2State instance;
    return instance;
  }

 private:
  bool is_warmup_{true};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
#endif
