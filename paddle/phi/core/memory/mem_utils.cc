// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/mem_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "paddle/common/enforce.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/memory/malloc.h"
#include "paddle/phi/core/memory/mem_visitor.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
namespace paddle {
namespace memory {

namespace {

std::vector<allocation::BlockPart> CollectPartsForAddressRange(
    const std::list<Block>& blocks, void* ptr, size_t size) {
  std::vector<allocation::BlockPart> out;
  if (size == 0) {
    return out;
  }

  const auto target_begin = reinterpret_cast<uintptr_t>(ptr);
  PADDLE_ENFORCE_LE(
      size,
      std::numeric_limits<uintptr_t>::max() - target_begin,
      common::errors::InvalidArgument(
          "Invalid VMM compact address range: ptr %p plus size %zu overflows.",
          ptr,
          size));
  const auto target_end = target_begin + size;
  size_t collected_len = 0;

  for (const auto& block : blocks) {
    const auto block_begin = reinterpret_cast<uintptr_t>(block.ptr_);
    const auto block_end = block_begin + block.size_;
    if (block_end <= target_begin) {
      continue;
    }
    if (block_begin >= target_end) {
      break;
    }

    const auto slice_begin = std::max(block_begin, target_begin);
    const auto slice_end = std::min(block_end, target_end);
    auto block_parts = allocation::SliceBlockPartsForRange(
        block.parts_, slice_begin - block_begin, slice_end - slice_begin);
    collected_len += slice_end - slice_begin;
    allocation::AppendBlockPartsTail(&out, &block_parts);
  }

  PADDLE_ENFORCE_EQ(
      collected_len,
      size,
      common::errors::InvalidArgument(
          "Invalid VMM compact address range: requested %zu bytes from ptr %p, "
          "but only collected %zu bytes.",
          size,
          ptr,
          collected_len));
  return out;
}

}  // namespace

bool IsContiguousAndAscending(const std::list<Block>& blocks) {
  return std::adjacent_find(
             blocks.begin(), blocks.end(), [](const Block& a, const Block& b) {
               return b.ptr_ < a.ptr_ ||
                      static_cast<uint8_t*>(a.ptr_) + a.size_ != b.ptr_;
             }) == blocks.end();
}

bool HasOverlap(size_t block_size, size_t remain_size) {
  return block_size > remain_size;
}

size_t TotalMemoryCompactor::Compact(std::list<Block>& blocks,
                                     void* start_ptr,
                                     void* end_ptr /*not used*/) {
#ifndef PADDLE_WITH_CUDA
  return -1;
#else
  if (!IsContiguousAndAscending(blocks)) return -1;
  void* new_ptr = start_ptr;
  size_t remaining_size = 0;
  std::list<Block> new_blocks;
  cudaDeviceSynchronize();
  for (auto& block : blocks) {
    if (!block.is_free_) {
      auto new_parts =
          CollectPartsForAddressRange(blocks, new_ptr, block.size_);
      if (block.ptr_ != new_ptr && block.ptr_ >= start_ptr) {
        auto src = static_cast<uint8_t*>(block.ptr_);
        auto dst = static_cast<uint8_t*>(new_ptr);
        auto sz = block.size_;

        if (HasOverlap(sz, remaining_size)) {
          for (size_t offset = 0; offset < sz; offset += remaining_size) {
            size_t current_chunk = std::min(remaining_size, sz - offset);
            cudaError_t err = cudaMemcpyAsync(dst + offset,
                                              src + offset,
                                              current_chunk,
                                              cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) return -1;
          }
        } else {
          cudaError_t err =
              cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice);
          if (err != cudaSuccess) return -1;
        }
      }
      block.allocation_->set_ptr(new_ptr);
      auto new_block = new_blocks.insert(
          new_blocks.end(), {new_ptr, block.size_, false, block.allocation_});
      new_block->parts_ = std::move(new_parts);
      block.allocation_->block_it_ = new_block;
      new_ptr = static_cast<uint8_t*>(new_ptr) + block.size_;
    } else {
      remaining_size += block.size_;
    }
  }
  cudaDeviceSynchronize();
  if (remaining_size > 0) {
    auto free_parts =
        CollectPartsForAddressRange(blocks, new_ptr, remaining_size);
    new_blocks.push_back({new_ptr, remaining_size, true});
    new_blocks.back().parts_ = std::move(free_parts);
  }

  blocks = std::move(new_blocks);
  return remaining_size;
#endif
}

#if defined(PADDLE_WITH_CUDA)
std::pair<size_t, size_t> VmmMaxFreeSize(const GPUPlace& place, int32_t n) {
  FreeMemoryMetricsVisitor free_memory_metrics_visitor(n);
  allocation::AllocatorFacade::Instance().Accept(place,
                                                 &free_memory_metrics_visitor);
  return std::make_pair(free_memory_metrics_visitor.GetLargeSize(),
                        free_memory_metrics_visitor.GetSumSize());
}

bool TryAllocBatch(const GPUPlace& place, const std::vector<size_t>& sizes) {
  TryAllocVisitor try_alloc_visitor(sizes);
  allocation::AllocatorFacade::Instance().Accept(place, &try_alloc_visitor);
  return try_alloc_visitor.IsTryAllocSuccess();
}

size_t VmmCompact(const GPUPlace& place) { return memory::Compact(place); }

std::vector<std::vector<std::pair<size_t, uintptr_t>>>
FreeBlockInfoOfVmmAllocator(const GPUPlace& place) {
  VMMFreeBlocksInfoVisitor free_blocks_info_visitor;
  allocation::AllocatorFacade::Instance().Accept(place,
                                                 &free_blocks_info_visitor);
  return free_blocks_info_visitor.GetFreeBlocksInfo();
}

std::vector<std::vector<std::tuple<size_t, uintptr_t, bool>>>
AllBlockInfoOfAllocator(const GPUPlace& place) {
  AllBlocksInfoVisitor all_blocks_info_visitor;
  allocation::AllocatorFacade::Instance().Accept(place,
                                                 &all_blocks_info_visitor);
  return all_blocks_info_visitor.GetAllBlocksInfo();
}

std::vector<std::tuple<uintptr_t, bool, uint64_t, size_t, int64_t, int64_t>>
GetAllocateEvent(const GPUPlace& place) {
  VMMAllocateRecordEventsVisitor allocate_record_event_visitor;
  allocation::AllocatorFacade::Instance().Accept(
      place, &allocate_record_event_visitor);
  return allocate_record_event_visitor.GetAllocateRecordEvents();
}

std::vector<size_t> GetCompactSize(const GPUPlace& place) {
  VMMAllocateCompactSizeVisitor allocate_compact_visitor;
  allocation::AllocatorFacade::Instance().Accept(place,
                                                 &allocate_compact_visitor);
  return allocate_compact_visitor.GetCompactSize();
}

#endif

}  // namespace memory
}  // namespace paddle
