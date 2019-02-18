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

#pragma once

#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/memory/detail/memory_block.h"
#include "paddle/fluid/memory/detail/system_allocator.h"
#include "paddle/fluid/platform/assert.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace memory {
namespace detail {

const int kDefaultBlockSize = 1 << 8;

struct BlockDesc {
  explicit BlockDesc(size_t size, void* left_buddy = nullptr)
      : size_(size), left_buddy_(left_buddy) {}

  size_t size_;
  void* left_buddy_;
};

class BudAllocator {
 public:
  // log2 function for ineger
  // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
  inline static size_t GetLevel(size_t size) {
    unsigned int r;  // result of log2(v) will go here
    unsigned int shift;

    r = (size > 0xFFFF) << 4;
    size >>= r;
    shift = (size > 0xFF) << 3;
    size >>= shift;
    r |= shift;
    shift = (size > 0xF) << 2;
    size >>= shift;
    r |= shift;
    shift = (size > 0x3) << 1;
    size >>= shift;
    r |= shift;
    r |= (size >> 1);
    return r;
  }

  // round up an integer to the nearest power of 2 which is larger than it
  // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  inline static size_t Normalize(size_t size) {
    size += (0 == size);
    --size;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    ++size;
    return size;
  }

 public:
  BudAllocator(std::unique_ptr<SystemAllocator> system_allocator,
               size_t min_block_size = kDefaultBlockSize)
      : system_allocator_(system_allocator.release()),
        min_block_size_(min_block_size) {}

  ~BudAllocator() {}

 public:
  void InitByLevel(size_t level);
  void InitBySize(size_t size);

  void* Alloc(size_t size);
  void Free(void* ptr);

  size_t FreeSize() { return (1 << max_level_) - total_used_; }
  size_t UsedSize() { return total_used_; }
  size_t NumOfBlocks() { return metainfo.size(); }

 public:
  // Disable copy and assignment
  BudAllocator(const BudAllocator&) = delete;
  BudAllocator& operator=(const BudAllocator&) = delete;

 private:
  void Split(const size_t& level);

 private:
  size_t total_used_ = 0;  // the total size of used memory

  size_t min_chunk_size_;  // the minimum size of each chunk

 private:
  /*! Allocate CPU/GPU memory from system */
  std::unique_ptr<SystemAllocator> system_allocator_;
  std::mutex mutex_;

 private:
  // memory pool to store available memory blocks
  std::vector<std::set<void*>> pool;

  // store the descrption information for allocatd memories
  std::unordered_map<void*, BlockDesc*> metainfo;

  // the desction information will also consume some memory
  // if the requested memory size is too small, then the
  // description consumption will even larger than it, this is meaningless
  size_t min_block_size_;

  // the max level of the buddy allocator
  // (1 << max_level_) is the total allocated memory size of the allocator
  size_t max_level_;
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
