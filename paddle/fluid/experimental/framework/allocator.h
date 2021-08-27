/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <cstdint>
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace experimental {
namespace framework {

class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual void* Allocate(size_t n) = 0;
  virtual void Deallocate(void* ptr, size_t n) = 0;
  virtual const platform::Place& place() const = 0;
};

inline void* Allocate(const std::shared_ptr<Allocator>& a, size_t n) {
  CHECK(a);
  return a->Allocate(n);
}
}  // namespace framework
}  // namespace experimental
}  // namespace paddle
