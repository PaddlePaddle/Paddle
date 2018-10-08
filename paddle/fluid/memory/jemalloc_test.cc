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

#include <gtest/gtest.h>
#include <jemalloc/jemalloc.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>

template <typename MallocFunc, typename... Args>
std::unique_ptr<void, void (*)(void *)> je_malloc_func(MallocFunc &&func,
                                                       Args... args) {
  return std::unique_ptr<void, void (*)(void *)>(
      func(std::forward<Args>(args)...), je_free);
}

TEST(jemalloc, jemalloc_not_overwrite_std) {
  ASSERT_TRUE(malloc != je_malloc);
  ASSERT_TRUE(calloc != je_calloc);
  ASSERT_TRUE(aligned_alloc != je_aligned_alloc);
  ASSERT_TRUE(realloc != je_realloc);

#ifdef _WIN32
  ASSERT_TRUE(_aligned_malloc != je_aligned_alloc);
#else
  ASSERT_TRUE(posix_memalign != je_posix_memalign);
#endif

  ASSERT_TRUE(free != je_free);
}

size_t n = (1 << 20);
size_t kAlignment = 64;

TEST(jemalloc, je_malloc) {
  auto x = je_malloc_func(je_malloc, n);
  ASSERT_NE(x, nullptr);
}

TEST(jemalloc, je_calloc) {
  using DataType = float;
  auto x = je_malloc_func(je_calloc, n, sizeof(DataType));
  ASSERT_NE(x, nullptr);
  bool is_all_zero = std::all_of(
      static_cast<DataType *>(x.get()), static_cast<DataType *>(x.get()) + n,
      [](const DataType &val) { return val == static_cast<DataType>(0); });
  ASSERT_TRUE(is_all_zero);
}

TEST(jemalloc, je_aligned_alloc) {
  auto x = je_malloc_func(je_aligned_alloc, kAlignment, n - 3);
  ASSERT_NE(x, nullptr);
  ASSERT_TRUE(reinterpret_cast<uintptr_t>(x.get()) % kAlignment == 0);
}

TEST(jemalloc, je_realloc) {
  {
    auto x = je_malloc_func(je_realloc, nullptr, n);
    ASSERT_NE(x, nullptr);
  }

  {
    auto x = je_malloc_func(je_malloc, n);
    ASSERT_NE(x, nullptr);
    auto y = je_malloc_func(je_realloc, x.release(), n / 2);
    ASSERT_NE(y, nullptr);
    auto z = je_malloc_func(je_realloc, y.release(), 2 * n);
    ASSERT_NE(z, nullptr);
  }
}

#ifndef _WIN32
inline void *je_posix_memalign_wrap(size_t alignment, size_t size) {
  void *ptr = nullptr;
  je_posix_memalign(&ptr, alignment, size);
  return ptr;
}

TEST(jemalloc, je_posix_memalign) {
  auto x = je_malloc_func(je_posix_memalign_wrap, kAlignment, n - 3);
  ASSERT_TRUE(reinterpret_cast<uintptr_t>(x.get()) % kAlignment == 0);
  ASSERT_NE(x, nullptr);
}
#endif
