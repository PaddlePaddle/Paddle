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

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/experimental/framework/tests/allocator.h"
#include "paddle/fluid/experimental/framework/tests/random.h"
#include "paddle/fluid/framework/generator.h"

namespace paddle {
namespace experimental {
namespace framework {
namespace tests {

template <typename T>
bool host_allocator_test(size_t vector_size) {
  std::vector<T> src(vector_size);
  std::generate(src.begin(), src.end(), make_generator(src));
  std::vector<T, CustomAllocator<T>> dst(
      src.begin(), src.end(),
      CustomAllocator<T>(std::make_shared<HostAllocatorSample>()));
  return std::equal(src.begin(), src.end(), dst.begin());
}

TEST(allocator, host) {
  CHECK(host_allocator_test<int8_t>(1000));
  CHECK(host_allocator_test<float>(1000));
  CHECK(host_allocator_test<int32_t>(1000));
  CHECK(host_allocator_test<int64_t>(1000));
}

}  // namespace tests
}  // namespace framework
}  // namespace experimental
}  // namespace paddle
