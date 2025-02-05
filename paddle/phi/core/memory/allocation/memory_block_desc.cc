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

#include <cstddef>
#include <functional>

#include "paddle/phi/core/memory/allocation/memory_block.h"

namespace paddle::memory::detail {

MemoryBlock::Desc::Desc(MemoryBlock::Type t,
                        size_t i,
                        size_t s,
                        size_t ts,
                        MemoryBlock* l,
                        MemoryBlock* r)
    : type(t),
      index(i),
      size(s),
      total_size(ts),
      left_buddy(l),
      right_buddy(r) {}

MemoryBlock::Desc::Desc()
    : type(MemoryBlock::INVALID_CHUNK),
      index(0),
      size(0),
      total_size(0),
      left_buddy(nullptr),
      right_buddy(nullptr) {}

namespace {

template <class T>
inline void hash_combine(std::size_t* seed, const T& v) {
  std::hash<T> hasher;
  (*seed) ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);
}

inline size_t hash(const MemoryBlock::Desc& metadata, size_t initial_seed) {
  size_t seed = initial_seed;

  hash_combine(&seed, static_cast<size_t>(metadata.type));
  hash_combine(&seed, metadata.index);
  hash_combine(&seed, metadata.size);
  hash_combine(&seed, metadata.total_size);
  hash_combine(&seed, metadata.left_buddy);
  hash_combine(&seed, metadata.right_buddy);

  return seed;
}

}  // namespace

void MemoryBlock::Desc::UpdateGuards() {
  guard_begin = hash(*this, 1);
  guard_end = hash(*this, 2);
}

bool MemoryBlock::Desc::CheckGuards() const {
  return guard_begin == hash(*this, 1) && guard_end == hash(*this, 2);
}

}  // namespace paddle::memory::detail
