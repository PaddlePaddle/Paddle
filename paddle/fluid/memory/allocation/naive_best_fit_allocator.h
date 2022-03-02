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

#pragma once
#include <stdint.h>
#include <algorithm>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

class NaiveBestFitAllocator : public Allocator {
 public:
  explicit NaiveBestFitAllocator(const platform::Place &p) : place_(p) {}

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  phi::Allocation *AllocateImpl(size_t size) override;
  void FreeImpl(phi::Allocation *allocation) override;
  uint64_t ReleaseImpl(const platform::Place &place) override;

 private:
  platform::Place place_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
