// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "gflags/gflags.h"
#include "paddle/fluid/platform/enforce.h"

DEFINE_string(
    allocator_strategy, "legacy",
    "The allocation strategy. Legacy means the original allocator of Fluid."
    "naive_best_fit means the experimental best fit allocator. "
    "auto_growth_best_fit means the experimental auto growth best fit "
    "allocator. Enum in [legacy, naive_best_fit, auto_growth_best_fit]");

namespace paddle {
namespace memory {
namespace allocation {

static AllocatorStrategy GetStrategyFromFlag() {
  if (FLAGS_allocator_strategy == "legacy") {
    return AllocatorStrategy::kLegacy;
  } else if (FLAGS_allocator_strategy == "naive_best_fit") {
    return AllocatorStrategy::kNaiveBestFit;
  } else if (FLAGS_allocator_strategy == "auto_growth_best_fit") {
    return AllocatorStrategy::kAutoGrowthBestFit;
  } else {
    PADDLE_THROW("Unsupported allocator strategy: %s",
                 FLAGS_allocator_strategy);
  }
}

AllocatorStrategy GetAllocatorStrategy() {
  static AllocatorStrategy strategy = GetStrategyFromFlag();
  return strategy;
}

void UseAllocatorStrategyGFlag() {}
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
