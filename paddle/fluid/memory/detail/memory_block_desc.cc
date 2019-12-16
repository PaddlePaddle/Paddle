/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <functional>

#include "paddle/fluid/memory/detail/memory_block.h"

namespace paddle {
namespace memory {
namespace detail {

namespace {

template <class T>
inline void hash_combine(std::size_t* seed, const T& v) {
  std::hash<T> hasher;
  (*seed) ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);
}

inline size_t hash(const MemoryBlock& metadata, size_t initial_seed) {
  size_t seed = initial_seed;

  hash_combine(&seed, metadata.get_data());
  hash_combine(&seed, static_cast<size_t>(metadata.get_type()));
  hash_combine(&seed, metadata.get_index());
  hash_combine(&seed, metadata.get_size());
  hash_combine(&seed, metadata.get_left_buddy());
  hash_combine(&seed, metadata.get_right_buddy());

  return seed;
}

}  // namespace

void MemoryBlock::UpdateGuards() {
  // #ifdef PADDLE_WITH_TESTING
  //   guard_begin = hash(*this, 1);
  //   guard_end = hash(*this, 2);
  // #endif
}

bool MemoryBlock::CheckGuards() const {
  // #ifdef PADDLE_WITH_TESTING
  //   return guard_begin == hash(*this, 1) && guard_end == hash(*this, 2);
  // #else
  return true;
  // #endif
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
