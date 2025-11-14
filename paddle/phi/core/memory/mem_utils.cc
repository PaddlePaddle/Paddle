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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
namespace paddle {
namespace memory {

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
      block.allocation_->block_it_ = new_blocks.insert(
          new_blocks.end(), {new_ptr, block.size_, false, block.allocation_});
      new_ptr = static_cast<uint8_t*>(new_ptr) + block.size_;
    } else {
      remaining_size += block.size_;
    }
  }
  cudaDeviceSynchronize();
  if (remaining_size > 0) {
    new_blocks.push_back({new_ptr, remaining_size, true});
  }

  blocks = std::move(new_blocks);
  return remaining_size;
#endif
}

}  // namespace memory
}  // namespace paddle
