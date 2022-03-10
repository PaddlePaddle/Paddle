/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/allocator.h"

namespace infrt {
namespace backends {

class CpuPhiAllocator : public phi::Allocator {
 public:
  static void deleter(phi::Allocation* ptr) { ::operator delete(ptr); }

  AllocationPtr Allocate(size_t bytes_size) {
    return AllocationPtr(
        new phi::Allocation(::operator new(bytes_size),
                            bytes_size,
                            phi::Place(phi::AllocationType::CPU)),
        deleter);
  }
};

}  // namespace backends
}  // namespace infrt
