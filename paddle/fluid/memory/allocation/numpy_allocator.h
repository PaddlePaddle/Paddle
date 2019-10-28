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
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class NumpyAllocation : public Allocation {
 public:
  NumpyAllocation(void* data_ptr, size_t size,
                  const std::function<void()>& deleter)
      : Allocation(data_ptr, size, platform::CPUPlace()) {
    _deleter = deleter;
  }

  void callDeleter() { _deleter(); }

 private:
  std::function<void()> _deleter;
};

class NumpyAllocator : public Allocator {
 public:
  NumpyAllocator(void* data_ptr, const std::function<void()>& deleter)
      : data_ptr(data_ptr), deleter(deleter) {}
  bool IsAllocThreadSafe() const override;

 protected:
  Allocation* AllocateImpl(size_t size) override;
  void FreeImpl(Allocation* allocation) override;

 private:
  void* data_ptr;
  const std::function<void()>& deleter;
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
