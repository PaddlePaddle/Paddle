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

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/api/lib/utils/storage.h"

namespace paddle {
namespace tests {

TEST(host_storage, external_stroage) {
  const size_t size{100};
  const auto a = std::make_shared<experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::intrusive_ptr<pten::Storage> in_storage =
      pten::make_intrusive<pten::TensorStorage>(a, size);
  char* data = static_cast<char*>(in_storage->data());
  for (size_t i = 0; i < size; ++i) {
    data[i] = i;
  }
  const size_t delta{1};
  const size_t n{10};
  auto ex_storage =
      pten::make_intrusive<experimental::ExternalStorage>(in_storage, delta, n);
  CHECK_EQ(ex_storage->size(), n);
  CHECK(paddle::platform::is_cpu_place(ex_storage->place()));
  CHECK(!ex_storage->OwnsMemory());
  for (size_t i = delta; i < delta + n; ++i) {
    CHECK_EQ(data[i], static_cast<char>(i));
  }
}

TEST(host_storage, external_vector) {
  std::vector<char> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }
  const size_t delta{1};
  const size_t n{10};
  auto ex_storage = pten::make_intrusive<experimental::ExternalStorage>(
      data.data(), n, paddle::platform::CPUPlace());
  CHECK_EQ(ex_storage->size(), n);
  CHECK(paddle::platform::is_cpu_place(ex_storage->place()));
  CHECK(!ex_storage->OwnsMemory());
  for (size_t i = delta; i < delta + n; ++i) {
    CHECK_EQ(data[i], static_cast<char>(i));
  }
}

}  // namespace tests
}  // namespace paddle
