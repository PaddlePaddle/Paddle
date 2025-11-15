// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"

#include <mutex>
#include "paddle/common/flags.h"

#include "paddle/phi/core/memory/allocation/aligned_allocator.h"

PHI_DEFINE_EXPORTED_bool(
    dump_vmm_allocation_info,
    false,
    "dump VirtualMemoryAutoGrowthBestFitAllocator's allocation info");
PHI_DEFINE_EXPORTED_bool(native_compact,
                         false,
                         "native_compact means compact memory after OOM, The "
                         "algorithm still needs to be upgraded.");

namespace paddle {
namespace memory {
namespace allocation {

bool NeedSplit(size_t block_size, size_t alignment, size_t alloc_size) {
  return block_size > (alloc_size * 2) || (block_size - alloc_size) > alignment;
}

VirtualMemoryAutoGrowthBestFitAllocator::
    VirtualMemoryAutoGrowthBestFitAllocator(
        const std::shared_ptr<Allocator> &underlying_allocator,
        size_t alignment,
        const phi::GPUPlace &place)
    : underlying_allocator_(
          std::make_shared<AlignedAllocator>(underlying_allocator, alignment)),
      alignment_(alignment),
      place_(place) {
  // NOTE(liujinnan): Only support TotalMemoryCompactor strategy for now.
  memory_compactor_ = std::make_unique<TotalMemoryCompactor>();
}

phi::Allocation *VirtualMemoryAutoGrowthBestFitAllocator::AllocateImpl(
    size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size = AlignedSize(size, alignment_);
  auto result = AllocFromFreeBlocks(size);

  if (!result) {
    ExtendOrCompact(size);
    result = AllocFromFreeBlocks(size);
  }

  return result;
}

void VirtualMemoryAutoGrowthBestFitAllocator::FreeImpl(
    phi::Allocation *allocation) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto block_it = static_cast<BlockAllocation *>(allocation)->block_it_;
  TryMergeBlock2Blocks(block_it);
  delete allocation;
}

void VirtualMemoryAutoGrowthBestFitAllocator::TryMergeBlock2Blocks(
    std::list<Block>::iterator block) {
  if (block->ptr_ == all_blocks_.front().ptr_ &&
      block->ptr_ == all_blocks_.back().ptr_) {
    block->is_free_ = true;
    free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
  } else if (block->ptr_ == all_blocks_.front().ptr_) {
    auto next = std::next(block);
    if (next->is_free_ &&
        reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ == next->ptr_) {
      // merge with next
      block->size_ += next->size_;
      block->is_free_ = true;
      free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
      all_blocks_.erase(next);
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    } else {
      block->is_free_ = true;
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    }
  } else if (block->ptr_ == all_blocks_.back().ptr_) {
    auto pre = std::prev(block);
    if (pre->is_free_ &&
        reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ == block->ptr_) {
      // merge with pre
      free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
      pre->size_ += block->size_;
      all_blocks_.erase(block);
      free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
    } else {
      block->is_free_ = true;
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    }
  } else {
    auto pre = std::prev(block);
    auto next = std::next(block);
    if (pre->is_free_ &&
        reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ == block->ptr_ &&
        !(next->is_free_ &&
          reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
              next->ptr_)) {
      // merge with pre
      free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
      pre->size_ += block->size_;
      all_blocks_.erase(block);
      free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
    } else if (next->is_free_ &&
               reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
                   next->ptr_ &&
               !(pre->is_free_ &&
                 reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ ==
                     block->ptr_)) {
      // merge with next
      block->size_ += next->size_;
      block->is_free_ = true;
      free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
      all_blocks_.erase(next);
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    } else if (pre->is_free_ &&
               reinterpret_cast<uint8_t *>(pre->ptr_) + pre->size_ ==
                   block->ptr_ &&
               next->is_free_ &&
               reinterpret_cast<uint8_t *>(block->ptr_) + block->size_ ==
                   next->ptr_) {
      // merge with pre and next
      free_blocks_.erase(std::make_pair(pre->size_, pre->ptr_));
      free_blocks_.erase(std::make_pair(next->size_, next->ptr_));
      pre->size_ += (block->size_ + next->size_);
      all_blocks_.erase(block);
      all_blocks_.erase(next);
      free_blocks_.emplace(std::make_pair(pre->size_, pre->ptr_), pre);
    } else {
      block->is_free_ = true;
      free_blocks_.emplace(std::make_pair(block->size_, block->ptr_), block);
    }
  }
}

std::optional<AllocationPtr>
VirtualMemoryAutoGrowthBestFitAllocator::AllocateOrCompact(size_t size) {
  AllocationPtr allocateptr = nullptr;
  // Just Allocate, no compact.
  if (!FLAGS_native_compact) {
    if (all_blocks_.empty()) {
      allocateptr = std::move(underlying_allocator_->Allocate(size));
    } else {
      auto free_block = std::prev(all_blocks_.end());
      if (free_block->is_free_) {
        assert(free_block->size_ < size);
        auto remain_size = size - free_block->size_;
        VLOG(1) << " Tail free block size {" << free_block->size_
                << "} is smaller than allocate size {" << size
                << "} after compact, re-alloc {" << remain_size << "}";
        allocateptr = std::move(underlying_allocator_->Allocate(remain_size));
      } else {
        VLOG(1) << "Tail block is not free, just allocate {" << size << "}";
        allocateptr = std::move(underlying_allocator_->Allocate(size));
      }
    }
    return allocateptr;
  }
  // Compact branch, try allocate and compact.
  try {
    allocateptr = std::move(underlying_allocator_->Allocate(size));
  } catch (const paddle::memory::allocation::BadAlloc &e) {
    VLOG(4) << "Do Memory Compact allocate size and compact " << size;
    size_t compact_free_size = memory_compactor_->Compact(
        all_blocks_, all_blocks_.front().ptr_, all_blocks_.back().ptr_);
    if (compact_free_size < 0) throw;
    VLOG(4) << "Memory Compacted Size: " << compact_free_size;
    auto free_block = std::prev(all_blocks_.end());
    if (free_block->is_free_ && free_block->size_ < size) {
      auto realloc_size = size - free_block->size_;
      VLOG(4) << "Free block size {" << free_block->size_
              << "} is smaller than allocate size {" << size
              << "} after compact, re-alloc {" << realloc_size << "}";
      try {
        auto realloc_ptr =
            underlying_allocator_->Allocate(size - free_block->size_);
        VLOG(4) << "Re-alloc size {" << realloc_ptr->size() << "} success";
        free_block->size_ += realloc_ptr->size();
        allocations_.push_back(std::move(realloc_ptr));  // hold allocation
      } catch (const paddle::memory::allocation::BadAlloc &e) {
        VLOG(4) << "Re-alloc size {" << realloc_size << "} failed";
        throw;
      }
    }
    return std::nullopt;
  }
  return allocateptr;
}

void VirtualMemoryAutoGrowthBestFitAllocator::ExtendOrCompact(size_t size) {
  void *alloc_ptr = nullptr;
  size_t alloc_size = 0;
  if (FLAGS_dump_vmm_allocation_info) {
    DumpInfo("===== Before ExtendOrCompact ===== request size: " +
             std::to_string(size));
  }

  auto allocateptr = AllocateOrCompact(size).value_or(nullptr);
  if (!allocateptr) {
    // Allocate failed and Compact success branch.
    free_blocks_.clear();
    auto free_block = std::prev(all_blocks_.end());
    if (free_block->is_free_) {
      free_blocks_.emplace(std::make_pair(free_block->size_, free_block->ptr_),
                           free_block);
    } else {
      LOG(INFO) << "Dont have free block after memory compact";
    }
    if (FLAGS_dump_vmm_allocation_info) {
      DumpInfo("===== After ExtendOrCompact do compact =====");
    }
    // After compact, Merge is not needed. just return.
    return;
  }

  alloc_ptr = allocateptr->ptr();
  alloc_size = allocateptr->size();
  allocations_.push_back(std::move(allocateptr));  // hold allocation

  if (all_blocks_.empty()) {
    all_blocks_.emplace_back(alloc_ptr, alloc_size, true);
    free_blocks_.emplace(std::make_pair(alloc_size, alloc_ptr),
                         all_blocks_.begin());
    return;
  }

  // insert to back
  auto block_it = all_blocks_.end();
  block_it--;
  if (block_it->is_free_ &&
      reinterpret_cast<uint8_t *>(block_it->ptr_) + block_it->size_ ==
          alloc_ptr) {
    // merge with pre
    free_blocks_.erase(std::make_pair(block_it->size_, block_it->ptr_));
    block_it->size_ += alloc_size;
    free_blocks_.emplace(std::make_pair(block_it->size_, block_it->ptr_),
                         block_it);
  } else {
    // do not merge
    all_blocks_.emplace_back(alloc_ptr, alloc_size, true);
    auto block_it = all_blocks_.end();
    block_it--;
    free_blocks_.emplace(std::make_pair(alloc_size, alloc_ptr), block_it);
  }
  if (FLAGS_dump_vmm_allocation_info) {
    DumpInfo("===== After ExtendOrCompact =====  request size: " +
             std::to_string(size) +
             " alloc size: " + std::to_string(alloc_size));
  }
}

phi::Allocation *VirtualMemoryAutoGrowthBestFitAllocator::AllocFromFreeBlocks(
    size_t size) {
  auto iter = free_blocks_.lower_bound(std::make_pair(size, nullptr));
  if (iter != free_blocks_.end()) {
    std::list<Block>::iterator block_it = iter->second;
    free_blocks_.erase(iter);
    if (NeedSplit(block_it->size_, alignment_, size)) {
      void *remaining_ptr = reinterpret_cast<uint8_t *>(block_it->ptr_) + size;
      size_t remaining_size = block_it->size_ - size;

      block_it->size_ = size;
      block_it->is_free_ = false;

      auto remaining_free_block = all_blocks_.insert(
          std::next(block_it), Block(remaining_ptr, remaining_size, true));
      free_blocks_.emplace(std::make_pair(remaining_size, remaining_ptr),
                           remaining_free_block);
    } else {
      block_it->is_free_ = false;
    }
    return new BlockAllocation(block_it, place_);
  }
  return nullptr;
}

void VirtualMemoryAutoGrowthBestFitAllocator::DumpInfo(
    std::string phase) const {
  size_t total = 0, free = 0, used = 0;
  std::cout << phase << std::endl;
  std::cout << "All_blocks_:" << std::endl;
  for (auto block = all_blocks_.begin(); block != all_blocks_.end(); ++block) {
    std::ostringstream oss_used;
    std::ostringstream oss_free;

    if (block->is_free_) {
      free += block->size_;
      oss_free << "(" << block->size_ << "," << block->ptr_ << ")";
    } else {
      used += block->size_;
      oss_used << "(" << block->size_ << "," << block->ptr_ << ","
               << block->allocation_->ptr() << ")";
    }

    std::cout << "is_free? " << block->is_free_ << "[" << oss_used.str()
              << "]\t[" << oss_free.str() << "]" << std::endl;
  }
  std::cout << total << "\t" << used << "\t" << free << std::endl;
  std::cout << "Free_blocks_:" << std::endl;
  for (const auto &[key, list_iter] : free_blocks_) {
    auto [size, ptr] = key;
    std::cout << "Size: " << size << ", Ptr: " << ptr << "\t" << list_iter->ptr_
              << std::endl;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
