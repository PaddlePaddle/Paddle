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
#include "paddle/phi/core/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"

namespace paddle {
namespace memory {

void AllocatorVisitor::Visit(RetryAllocator* allocator) {
  allocator->GetUnderLyingAllocator()->Accept(this);
}

void AllocatorVisitor::Visit(StatAllocator* allocator) {
  allocator->GetUnderLyingAllocator()->Accept(this);
}

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

void FreeMemoryMetricsVisitor::Visit(
    VirtualMemoryAutoGrowthBestFitAllocator* allocator) {
  std::tie(large_size_, sum_size_) =
      allocator->SumLargestFreeBlockSizes(nums_blocks_);
  VLOG(1) << "Visit VirtualMemoryAutoGrowthBestFitAllocator large_free_size:"
          << large_size_ << " sum_free_size:" << sum_size_;
}
}  // namespace memory
}  // namespace paddle
