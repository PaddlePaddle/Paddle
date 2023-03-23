// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <cstdint>
#include <cstdlib>

namespace ir {
std::size_t hash_combine(std::size_t lhs, std::size_t rhs);

void *aligned_malloc(size_t size, size_t alignment);

void aligned_free(void *mem_ptr);

inline void *offset_address(void *ptr, uint32_t offset) {
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) | offset);
}

inline uint32_t solve_offset(void *ptr, uint32_t max_offset) {
  return reinterpret_cast<uintptr_t>(ptr) & max_offset;
}

inline void *reset_offset_address(void *ptr_offset, uint32_t max_offset) {
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr_offset) &
                                  (~max_offset));
}

}  // namespace ir
