// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"

#include <algorithm>
#include <iterator>

#include "paddle/phi/core/enforce.h"
namespace paddle {
namespace memory {
namespace allocation {

namespace {

template <typename Map, typename Key, typename Value>
void EmplaceOrEnforce(Map* map,
                      Key&& key,
                      Value&& value,
                      const char* map_name) {
  const bool inserted =
      map->try_emplace(std::forward<Key>(key), std::forward<Value>(value))
          .second;
  PADDLE_ENFORCE_EQ(
      inserted,
      true,
      common::errors::AlreadyExists(
          "Duplicate key inserted into %s, allocator state is inconsistent.",
          map_name));
}

std::vector<BlockPartV2> SlicePartsForRange(
    const std::vector<BlockPartV2>& parts,
    size_t range_offset,
    size_t range_len) {
  // parts describes one logical block as an ordered list of handle slices.
  // The target range is also expressed in that logical block address space.
  //
  // Example:
  //   parts:        [part0 len=2][part1 len=3][part2 len=4]
  //   logical idx:   0          2            5            9
  //   range:              [------ range ------)
  //                       1                    7
  //
  //   result:
  //     - part0 contributes a right-side slice [1,2)
  //     - part1 is fully covered and copied as-is
  //     - part2 contributes a left-side slice [5,7)
  //
  // A block-level parts_ list always describes one contiguous logical range,
  // but each element may only cover a slice of its underlying handle. Scan the
  // logical range once, intersect each part with [range_offset, range_end),
  // and rebuild the sliced view in order.
  std::vector<BlockPartV2> sliced_parts;
  if (range_len == 0 || parts.empty()) {
    return sliced_parts;
  }

  sliced_parts.reserve(parts.size());
  const size_t range_end = range_offset + range_len;
  size_t cursor = 0;

  for (const auto& part : parts) {
    const size_t part_block_begin = cursor;
    const size_t part_block_end = cursor + part.len;
    cursor = part_block_end;

    if (part_block_end <= range_offset) {
      continue;
    }
    if (part_block_begin >= range_end) {
      break;
    }

    const size_t slice_begin = std::max(part_block_begin, range_offset);
    const size_t slice_end = std::min(part_block_end, range_end);
    auto slice =
        part.Slice(slice_begin - part_block_begin, slice_end - slice_begin);

    if (sliced_parts.empty() || !sliced_parts.back().TryExtend(slice)) {
      sliced_parts.push_back(std::move(slice));
    }
  }
  return sliced_parts;
}

std::vector<BlockPartV2> BuildBlockPartsFromHandleLayout(
    const HandleLayout& layout) {
  std::vector<BlockPartV2> parts;
  parts.reserve(layout.size());
  // The bottom allocator only returns the fixed-handle list for one raw
  // allocation. Best-fit turns that list into block-level slices that will
  // later be split/merged/remapped as BlockV2::parts_ evolves.
  for (const auto& handle : layout) {
    parts.push_back(BlockPartV2{handle, 0, handle->size});
  }
  return parts;
}

void AppendPartsTail(std::vector<BlockPartV2>* dst,
                     std::vector<BlockPartV2>* src) {
  // dst and src each describe one logical block. When merge joins two adjacent
  // FREE blocks, concatenate their parts_ while collapsing the boundary if it
  // happens to land in the middle of one handle. The source block is erased
  // right after merge, so its parts_ can be moved instead of copied.
  if (src->empty()) {
    return;
  }
  dst->reserve(dst->size() + src->size());
  auto begin = src->begin();
  if (!dst->empty() && dst->back().TryExtend(src->front())) {
    ++begin;
  }
  dst->insert(dst->end(),
              std::make_move_iterator(begin),
              std::make_move_iterator(src->end()));
}

}  // namespace

VMMAutoGrowthBestFitAllocatorV2::VMMAutoGrowthBestFitAllocatorV2(
    const std::shared_ptr<CUDAVirtualMemAllocatorV2>& underlying_allocator,
    size_t alignment,
    const GPUPlace& place,
    PoolType pool_type)
    : underlying_allocator_(underlying_allocator),
      alignment_(alignment),
      place_(place),
      pool_type_(pool_type) {}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocateImpl(size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  const size_t requested_size = AlignedSize(size, alignment_);
  if (auto* allocation = AllocFromFreeBlocks(requested_size)) {
    return allocation;
  }

  auto allocation = static_unique_ptr_cast<Allocation>(
      underlying_allocator_->Allocate(requested_size));
  HandleLayout layout;
  PADDLE_ENFORCE_EQ(underlying_allocator_->CollectAllocationHandleLayout(
                        allocation->ptr(), &layout),
                    true,
                    common::errors::NotFound(
                        "Can not collect VMM handle layout for allocation %p.",
                        allocation->ptr()));
  auto parts = BuildBlockPartsFromHandleLayout(layout);
  auto* raw_allocation = allocation.get();
  underlying_allocations_.emplace_back(std::move(allocation));
  const size_t raw_size = raw_allocation->size();
  // Grow first obtains one raw allocation from the bottom VMM provider, then
  // immediately converts it into block state. If the raw allocation is larger
  // than the requested size (for example due to handle-size rounding), split
  // it right away so the remainder becomes a reusable FREE block instead of
  // being hidden inside one oversized ACTIVE block.
  auto active_parts = SlicePartsForRange(parts, 0, requested_size);
  const size_t remaining_size = raw_size - requested_size;

  BlockV2 block;
  block.ptr_ = raw_allocation->ptr();
  block.size_ = requested_size;
  block.type_ = BlockType::kActive;
  block.parts_ = std::move(active_parts);
  block.pool_type_ = pool_type_;
  auto it = all_blocks_.insert(all_blocks_.end(), std::move(block));
  EmplaceOrEnforce(&allocated_blocks_, it->ptr_, it, "allocated_blocks_");

  if (remaining_size > 0) {
    BlockV2 remaining_block;
    remaining_block.ptr_ =
        reinterpret_cast<uint8_t*>(raw_allocation->ptr()) + requested_size;
    remaining_block.size_ = remaining_size;
    remaining_block.type_ = BlockType::kFree;
    remaining_block.parts_ =
        SlicePartsForRange(parts, requested_size, remaining_size);
    remaining_block.pool_type_ = pool_type_;
    auto remain_it =
        all_blocks_.insert(std::next(it), std::move(remaining_block));
    InsertFreeBlock(remain_it);
  }

  return new Allocation(it->ptr_, it->ptr_, it->size_, place_);
}

void VMMAutoGrowthBestFitAllocatorV2::FreeImpl(phi::Allocation* allocation) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto ptr = allocation->ptr();
  auto found = allocated_blocks_.find(ptr);
  PADDLE_ENFORCE_NE(
      found,
      allocated_blocks_.end(),
      common::errors::NotFound("Can not find active block for allocation %p in "
                               "VMMAutoGrowthBestFitAllocatorV2.",
                               ptr));
  auto it = found->second;
  allocated_blocks_.erase(it->ptr_);
  it->type_ = BlockType::kFree;
  TryMerge(it);
  delete allocation;
}

bool VMMAutoGrowthBestFitAllocatorV2::SetBlockRemapEvent(void* ptr,
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
                                                         gpuStream_t stream,
                                                         gpuEvent_t event
#else
                                                         void* stream,
                                                         void* event
#endif
) {
  std::lock_guard<SpinLock> guard(spinlock_);
  auto it = allocated_blocks_.find(ptr);
  if (it == allocated_blocks_.end()) {
    return false;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  it->second->last_use_stream_ = stream;
  it->second->remap_safe_event_ = event;
#else
  (void)stream;
  (void)event;
#endif
  return true;
}

phi::Allocation* VMMAutoGrowthBestFitAllocatorV2::AllocFromFreeBlocks(
    size_t size) {
  auto it = free_blocks_.lower_bound({size, nullptr});
  if (it == free_blocks_.end()) {
    return nullptr;
  }

  auto block_it = it->second;
  EraseFreeBlock(block_it);

  if (block_it->size_ > size) {
    const size_t remaining_size = block_it->size_ - size;
    BlockV2 remaining_block;
    remaining_block.ptr_ = reinterpret_cast<uint8_t*>(block_it->ptr_) + size;
    remaining_block.size_ = remaining_size;
    remaining_block.type_ = BlockType::kFree;
    remaining_block.parts_ =
        SlicePartsForRange(block_it->parts_, size, remaining_size);
    remaining_block.pool_type_ = block_it->pool_type_;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    // Inherit last_use_stream_ and remap_safe_event_ from the original block.
    //
    // Under fast-GC (same-stream reuse), StreamSafeCUDAAllocator hands this
    // FREE block back without waiting on the recorded event — CUDA stream
    // ordering alone guarantees that the NEW kernel (using the ACTIVE portion)
    // runs after the old one.  However, the REMAINING free portion is still
    // physically backed by memory the old kernel may still be touching.  If
    // the Compactor saw remap_safe_event_ == nullptr it would assume "never
    // used, safe to unmap" and cuMemUnmap while the old kernel is still
    // reading — causing a GPU fault.
    //
    // Inheriting the event is correct because the event was recorded AFTER the
    // last kernel that accessed the ENTIRE original block; the remaining
    // portion is a subset, so the same event guards it.
    //
    // For grow-split (AllocateImpl), the remaining block comes from freshly
    // allocated memory that was never used, so its event is naturally nullptr
    // — which correctly means "safe to remap".
    //
    // owning_stream_ is cleared: nobody "owns" a free fragment.
    remaining_block.owning_stream_ = nullptr;
    remaining_block.last_use_stream_ = block_it->last_use_stream_;
    remaining_block.remap_safe_event_ = block_it->remap_safe_event_;
#endif

    block_it->size_ = size;
    block_it->parts_ = SlicePartsForRange(block_it->parts_, 0, size);
    auto remain_it =
        all_blocks_.insert(std::next(block_it), std::move(remaining_block));
    InsertFreeBlock(remain_it);
  }

  block_it->type_ = BlockType::kActive;
  EmplaceOrEnforce(
      &allocated_blocks_, block_it->ptr_, block_it, "allocated_blocks_");
  return new Allocation(
      block_it->ptr_, block_it->ptr_, block_it->size_, place_);
}

void VMMAutoGrowthBestFitAllocatorV2::InsertFreeBlock(BlockListIt it) {
  EmplaceOrEnforce(
      &free_blocks_, std::make_pair(it->size_, it->ptr_), it, "free_blocks_");
}

void VMMAutoGrowthBestFitAllocatorV2::EraseFreeBlock(BlockListIt it) {
  free_blocks_.erase({it->size_, it->ptr_});
}

void VMMAutoGrowthBestFitAllocatorV2::TryMerge(BlockListIt it) {
  // Only adjacent FREE blocks are merged here. ACTIVE blocks are never touched,
  // and GAP blocks remain as explicit holes for later remap/GAP handling.
  // all_blocks_ is the full VA-ordered block list, so adjacency is checked
  // against neighboring entries in that list.
  if (it != all_blocks_.begin()) {
    auto prev = std::prev(it);
    if (prev->type_ == BlockType::kFree &&
        reinterpret_cast<uint8_t*>(prev->ptr_) + prev->size_ ==
            reinterpret_cast<uint8_t*>(it->ptr_)) {
      EraseFreeBlock(prev);
      AppendPartsTail(&prev->parts_, &it->parts_);
      prev->size_ += it->size_;
      all_blocks_.erase(it);
      it = prev;
    }
  }

  auto next = std::next(it);
  if (next != all_blocks_.end() && next->type_ == BlockType::kFree &&
      reinterpret_cast<uint8_t*>(it->ptr_) + it->size_ ==
          reinterpret_cast<uint8_t*>(next->ptr_)) {
    EraseFreeBlock(next);
    AppendPartsTail(&it->parts_, &next->parts_);
    it->size_ += next->size_;
    all_blocks_.erase(next);
  }

  InsertFreeBlock(it);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
