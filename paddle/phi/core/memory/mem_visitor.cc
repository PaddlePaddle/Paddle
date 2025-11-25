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

#include "paddle/phi/core/memory/mem_visitor.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/retry_allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/stat_allocator.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#endif

namespace paddle {
namespace memory {

void allocation::Allocator::Accept(AllocatorVisitor* visitor) {
  visitor->Visit(this);
}

void AllocatorVisitor::Visit(RetryAllocator* allocator) {
  allocator->GetUnderLyingAllocator()->Accept(this);
}

void AllocatorVisitor::Visit(StatAllocator* allocator) {
  allocator->GetUnderLyingAllocator()->Accept(this);
}

#ifdef PADDLE_WITH_CUDA
void AllocatorVisitor::Visit(StreamSafeCUDAAllocator* allocator) {
  const std::vector<StreamSafeCUDAAllocator*>& allocators =
      allocator->GetAllocatorByPlace();
  for (StreamSafeCUDAAllocator* allocator : allocators) {
    allocator->GetUnderLyingAllocator()->Accept(this);
  }
}

void AllocatorVisitor::Visit(
    VirtualMemoryAutoGrowthBestFitAllocator* allocator) {
  allocator->GetUnderLyingAllocator()->Accept(this);
}

void AllocatorVisitor::Visit(
    VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator* allocator) {
  allocator->GetSmallAllocator()->Accept(this);
  allocator->GetLargeAllocator()->Accept(this);
}

void AllocatorComputeStreamVisitor::Visit(StreamSafeCUDAAllocator* allocator) {
  const std::vector<StreamSafeCUDAAllocator*>& allocators =
      allocator->GetAllocatorByPlace();
  assert(!allocators.empty());
  // NOTE(liujinnan): Currently, the Allocator initialization sequence is as
  // follows: the compute stream Allocator is initialized at program startup,
  // and then, when multiple streams are encountered at runtime, additional
  // Allocators are created and added to the end of the `allocator_map_` in
  // `StreamSafeCUDAAllocator`. Therefore, we can use the first allocator in
  // `allocator_map_` as the compute stream allocator. Although this approach is
  // somewhat ugly and may not be robust, it is currently effective.
  allocators[0]->GetUnderLyingAllocator()->Accept(this);
}

void FreeMemoryMetricsVisitor::Visit(
    VirtualMemoryAutoGrowthBestFitAllocator* allocator) {
  auto [large_size, sum_size] =
      allocator->SumLargestFreeBlockSizes(nums_blocks_);
  large_size_ = std::max(large_size_, large_size);
  sum_size_ = std::max(sum_size_, sum_size);
}

void TryAllocVisitor::Visit(
    VirtualMemoryAutoGrowthBestFitAllocator* allocator) {
  // TODO(liujinnan): More detailed handling of multi-stream and MultiScalePool
  // scenarios.
  is_try_alloc_success_ |= allocator->TryAllocateBatch(sizes_);
}

void VMMFreeBlocksInfoVisitor::Visit(
    VirtualMemoryAutoGrowthBestFitAllocator* allocator) {
  std::vector<std::pair<size_t, uintptr_t>> keys;
  for (const auto& item : allocator->GetFreeBlocks()) {
    size_t size = item.first.first;
    uintptr_t addr = reinterpret_cast<uintptr_t>(item.first.second);
    keys.emplace_back(size, addr);
  }
  if (!keys.empty()) {
    free_blocks_info_.push_back(keys);
  }
}

void VMMAllBlocksInfoVisitor::Visit(
    VirtualMemoryAutoGrowthBestFitAllocator* allocator) {
  std::vector<std::tuple<size_t, uintptr_t, bool>> info;
  for (const auto& item : allocator->GetAllBlocks()) {
    size_t size = item.size_;
    uintptr_t addr = reinterpret_cast<uintptr_t>(item.ptr_);
    bool is_free = item.is_free_;
    info.emplace_back(size, addr, is_free);
  }
  if (!info.empty()) {
    all_blocks_info_.push_back(info);
  }
}

void VmmTensorPartsVisitor::Visit(
    VirtualMemoryAutoGrowthBestFitAllocator* allocator) {
  if (found_) {
    return;
  }
  std::vector<BlockPart> parts;
  if (allocator->CollectTensorParts(target_ptr_, &parts)) {
    found_ = true;
    parts_ = std::move(parts);
    return;
  }
  allocator->GetUnderLyingAllocator()->Accept(this);
}
#endif
}  // namespace memory
}  // namespace paddle
