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

#include <vector>

#include "gtest/gtest.h"

#include "paddle/pten/core/storage.h"
#include "paddle/pten/tests/core/allocator.h"

namespace pten {
namespace tests {

TEST(host_storage, internal) {
  // TODO(Shixiaowei02): Here we need to consider the case
  // where the size is zero.
  const size_t size{100};
  const auto a = std::make_shared<FancyAllocator>();
  TensorStorage storage(a, size);
  CHECK_EQ(storage.size(), size);
  CHECK(paddle::platform::is_cpu_place(storage.place()));
  CHECK(storage.OwnsMemory());
  CHECK(storage.allocator() == a);
  storage.Realloc(size + 100);
  CHECK_EQ(storage.size(), size + 100);
}

}  // namespace tests
}  // namespace pten
