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

#include "paddle/fluid/memory/detail/bud_allocator.h"
#include <algorithm>
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace detail {

void BudAllocator::InitByLevel(size_t level) {
  size_t size = 1 << level;
  max_level_ = level;

  pool.resize(level + 1);
  VLOG(10) << "allocate init size : " << size;

  size_t index = 0;
  void *ptr = system_allocator_->Alloc(&index, size);

  pool[level].insert(ptr);
  metainfo[ptr] = new BlockDesc(size);
}

void BudAllocator::InitBySize(size_t size) {
  VLOG(10) << "request init size : " << size;
  size = Normalize(size);
  auto level = GetLevel(size);
  InitByLevel(level);
}

// split large block into small pieces
void BudAllocator::Split(const size_t &level) {
  auto addr = *pool[level].begin();
  pool[level].erase(pool[level].begin());

  auto ll = level - 1;
  auto size = 1 << ll;
  pool[ll].insert(addr);

  auto right = static_cast<void *>(static_cast<char *>(addr) + size);
  pool[ll].insert(right);

  // update description info
  metainfo[addr]->size_ = size;
  metainfo[right] = new BlockDesc(size, addr);
}

void *BudAllocator::Alloc(size_t size) {
  // Acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  VLOG(10) << "request allocate : " << size;

  size = std::max(Normalize(size), min_block_size_);
  VLOG(10) << "actually allocate : " << size;
  auto target_level = GetLevel(size);

  auto level = target_level;
  while (level <= max_level_) {
    if (pool[level].size()) break;
    ++level;
  }

  if (level > max_level_) {
    VLOG(10) << "No available memory in BudAllocator";
    return nullptr;
  }

  if (level > target_level) {
    // split current level
    while (level > target_level) {
      // split the first node of current level
      Split(level);
      --level;
    }
  }

  auto addr = *pool[level].begin();
  pool[level].erase(pool[level].begin());

  total_used_ += size;
  return addr;
}

void BudAllocator::Free(void *addr) {
  // Acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  PADDLE_ENFORCE(metainfo.find(addr) != metainfo.end(),
                 "unexpected status, cannot find meta info of pointer : %p",
                 addr);
  total_used_ -= metainfo[addr]->size_;

  // merge buddy blocks recursively
  for (;;) {
    bool flags = true;

    auto level = GetLevel(metainfo[addr]->size_);
    auto left_buddy = metainfo[addr]->left_buddy_;

    if (left_buddy) {
      PADDLE_ENFORCE(metainfo.find(left_buddy) != metainfo.end(),
                     "unexpected status, cannot find meta info of pointer : %p",
                     left_buddy);
    }
    // left_buddy == nullptr means this is the first block
    // else if (!left_buddy)
    //   VLOG(11) << "meet first block address : " << addr << endl;

    // merge left_buddy if it's not IN_USE and has equal size
    auto pos = pool[level].find(left_buddy);
    if (pos != pool[level].end()) {
      metainfo[left_buddy]->size_ <<= 1;

      auto iter = metainfo.find(addr);
      delete iter->second;
      metainfo.erase(iter);

      pool[level].erase(pos);
      // pool[level + 1].insert(left_buddy);

      // update addr
      addr = left_buddy;

      flags = false;
    }

    // merge right_buddy if it's not IN_USE and it's the 'right buddy' -
    // current block is the left buddy of the right_buddy
    auto right_buddy =
        static_cast<void *>(static_cast<char *>(addr) + metainfo[addr]->size_);
    pos = pool[level].find(right_buddy);
    if (pos != pool[level].end()) {
      auto iter = metainfo.find(right_buddy);

      if (iter->second->left_buddy_ == addr) {
        metainfo[addr]->size_ <<= 1;
        delete iter->second;
        metainfo.erase(iter);

        pool[level].erase(pos);

        flags = false;
      }
    }

    // no free buddy blocks found, break the merge process
    if (flags) break;
  }

  // insert addr back to pool
  auto level = GetLevel(metainfo[addr]->size_);
  pool[level].insert(addr);

  return;
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
