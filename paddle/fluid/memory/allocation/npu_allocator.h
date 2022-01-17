// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <mutex>  // NOLINT
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

class NPUAllocator : public Allocator {
 public:
  explicit NPUAllocator(const platform::NPUPlace& place) : place_(place) {}

  bool IsAllocThreadSafe() const override;

 protected:
  void FreeImpl(pten::Allocation* allocation) override;
  pten::Allocation* AllocateImpl(size_t size) override;

 private:
  platform::NPUPlace place_;
  std::once_flag once_flag_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
