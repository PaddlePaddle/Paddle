// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/spin_lock.h"

namespace paddle {
namespace memory {
namespace allocation {

class AutoGrowthBestFitAllocatorV2 : public AutoGrowthBestFitAllocator {
 public:
  AutoGrowthBestFitAllocatorV2(
      const std::shared_ptr<Allocator> &underlying_allocator,
      size_t alignment,
      platform::CUDAPlace place,
      size_t chunk_size = 0,
      bool allow_free_idle_chunk = true,
      int extra_padding_size = 0);

 protected:
  phi::Allocation *AllocateImpl(size_t size) override;

 private:
  bool is_strict_matching_state_;
  platform::CUDAPlace place_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
