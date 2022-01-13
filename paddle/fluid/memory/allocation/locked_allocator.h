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
#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT

#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

// A allocator to make underlying allocator thread safe.
class LockedAllocator : public Allocator {
 public:
  explicit LockedAllocator(std::shared_ptr<Allocator> underlying_allocator);
  bool IsAllocThreadSafe() const override;

 protected:
  void FreeImpl(pten::Allocation *allocation) override;
  pten::Allocation *AllocateImpl(size_t size) override;

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
  std::unique_ptr<std::mutex> mtx_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
