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

#pragma once

#include <memory>

#include "paddle/fluid/memory/memory_block2.h"
#include "paddle/fluid/memory/place.h"

namespace paddle {
namespace memory {
namespace detail {

template <typename Place>
class BuddyAllocator {
 public:
  BuddyAllocator(Place, uint64_t min_block_size, uint64_t max_block_size);
  void* Alloc(uint64_t unaligned_size);
  void Free(void* ptr);
  uint64_t Used() const;

 private:
  uint64_t AlignSize(uint64_t unaligned_size, uint64_t alignment) const;
  MemoryBlock* FindBuddy(MemoryBlock* p, uint64_t size);

 private:
  Place place_;
  const uint64_t kMinBlockSize;
  const uint64_t kMaxBlockSize;
  // MemoryBlock* root_;
  std::unique_ptr<MemoryBlock> root_;
  uint64_t total_size_;
  uint64_t total_used_;
  uint64_t current_block_size_;
  std::mutex mutex_;
  DISABLE_COPY_AND_ASSIGN(BuddyAllocator);
};

template <typename Place>
BuddyAllocator::BuddyAllocator(Place, uint64_t min_block_size,
                               uint64_t max_block_size)
    : place_(place),
      kMinBlockSize(min_block_size),
      kMaxBlockSize(max_block_size),
      root_(nullptr),
      total_size_(0UL),
      total_used_(0UL) {}

template <typename Place>
void* BuddyAllocator::Alloc(uint64_t unaligned_size) {
  if (unaligned_size == 0) return nullptr;
  auto size = AlignSize(unaligned_size + sizeof(BuddyAllocator), kMinBlockSize);

  std::unique_lock<std::mutex> lock(mutex_);
  auto* buddy = FindBuddy(root_);

  if (root_ == nullptr) {
    void* p = Alloc(size);
    PADDLE_ENFORCE(p != nullptr, "Failed allocate memory from system.")
  }
}

template <typename Place>
uint64_t BuddyAllocator::AlignSize(uint64_t unaligned_size,
                                   uint64_t alignment) const {
  return unaligned_size % alignment == 0
             ? unaligned_size
             : (unaligned_size / alignment + 1) * alignment;
}

template <typename Place>
MemoryBlock* FindBuddy(MemoryBlock* p, uint64_t size) {
  MemoryBlock* ret = nullptr;
  if (p->Size() == kMinBlockSize && size < p->Size()) {
    ret = p;
  }
  if (ret == nullptr) {
    ret = FindBuddy(p->LeftBuddy(), size);
  }
  if (ret == nullptr) {
    ret = FindBuddy(p->RightBuddy(), size);
  }
  return ret;
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
