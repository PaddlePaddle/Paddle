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
#include "paddle/fluid/framework/generator.h"
#include "paddle/pten/tests/core/allocator.h"
#include "paddle/pten/tests/core/random.h"
#include "paddle/pten/tests/core/timer.h"

namespace pten {
namespace tests {

template <typename T>
bool host_allocator_test(size_t vector_size) {
  std::vector<T> src(vector_size);
  std::generate(src.begin(), src.end(), make_generator(src));
  std::vector<T, CustomAllocator<T>> dst(
      src.begin(),
      src.end(),
      CustomAllocator<T>(std::make_shared<HostAllocatorSample>()));
  return std::equal(src.begin(), src.end(), dst.begin());
}

TEST(raw_allocator, host) {
  CHECK(host_allocator_test<float>(1000));
  CHECK(host_allocator_test<int32_t>(1000));
  CHECK(host_allocator_test<int64_t>(1000));
}

class StorageRawAlloc {
 public:
  StorageRawAlloc(const std::shared_ptr<RawAllocator>& a, size_t size)
      : alloc_(a) {
    data_ = alloc_->Allocate(size);
  }
  ~StorageRawAlloc() { alloc_->Deallocate(data_, size); }

 private:
  void* data_;
  size_t size;
  std::shared_ptr<RawAllocator> alloc_;
};

class StorageFancyAlloc {
 public:
  StorageFancyAlloc(const std::shared_ptr<Allocator>& a, size_t size)
      : alloc_(a), allocation_(a->Allocate(size)) {}

 private:
  std::shared_ptr<Allocator> alloc_;
  Allocation allocation_;
};

TEST(benchmark, allocator) {
  std::shared_ptr<RawAllocator> raw_allocator(new HostAllocatorSample);
  std::shared_ptr<Allocator> fancy_allocator(new FancyAllocator);
  const size_t cycles = 100;
  Timer timer;
  double t1{}, t2{};
  for (size_t i = 0; i < cycles; ++i) {
    timer.tic();
    for (size_t i = 0; i < cycles; ++i) {
      StorageRawAlloc(raw_allocator, i * 100);
    }
    t1 += timer.toc();
    timer.tic();
    for (size_t i = 0; i < cycles; ++i) {
      StorageFancyAlloc(fancy_allocator, i * 100);
    }
    t2 += timer.toc();
  }
  std::cout << "The cost of raw alloc is " << t1 << "ms.\n";
  std::cout << "The cost of fancy alloc with place is " << t2 << "ms.\n";
}

}  // namespace tests
}  // namespace pten
