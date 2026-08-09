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

#include "paddle/phi/core/memory/allocation/free_block_remap_compactor.h"

#if defined(PADDLE_WITH_CUDA)

#include <utility>
#include <vector>

#include "glog/logging.h"

namespace paddle {
namespace memory {
namespace allocation {

size_t FreeBlockRemapCompactor::Compact(
    std::list<BlockV2>* blocks,
    size_t requested_size,
    const RemapTransaction::SourcePages& source_pages,
    RemapTransaction::DestinationPolicy destination_policy) {
  const size_t handle_size = vmm_allocator_->handle_size();
  RemapTransaction transaction(vmm_allocator_.get(),
                               handle_size,
                               commit_destination_allocations_,
                               can_prepare_destination_range_,
                               prepare_destination_range_);

  // Clear any sticky CUDA error before we start.
  cudaGetLastError();
  // No cudaDeviceSynchronize here. Source collection uses per-event query
  // inside RemapTransaction to avoid a full pipeline stall.

  // Roll back transaction state before propagating unexpected allocator or
  // CUDA failures. Expected inability to compact is returned in the result.
  try {
    auto compact_result = transaction.CompactFreeBlocks(
        blocks, requested_size, pool_type_, source_pages, destination_policy);
    const auto& stats = compact_result.source_stats;

    auto compact_cuda_err = cudaPeekAtLastError();
    PADDLE_ENFORCE_EQ(
        compact_cuda_err,
        cudaSuccess,
        common::errors::External(
            "CUDA error detected immediately after VMM V2 compact: %s. "
            "pool=%d requested=%zu remapped_handles=%zu remapped_bytes=%zu "
            "move_commit_us=%zu free_blocks=%zu safe_blocks=%zu "
            "fully_covered=%zu event_blocked=%zu remapped_blocked=%zu "
            "unknown_safety_blocked=%zu",
            cudaGetErrorString(compact_cuda_err),
            static_cast<int>(pool_type_),
            requested_size,
            compact_result.remapped_handle_count,
            compact_result.remapped_bytes,
            static_cast<size_t>(compact_result.move_commit_us),
            stats.free_block_count,
            stats.safe_block_count,
            stats.fully_covered_count,
            stats.event_blocked_count,
            stats.remapped_blocked_count,
            stats.unknown_safety_blocked_count));
    if (compact_result.remapped_handle_count == 0) {
      return 0;
    }

    if (!compact_result.success) {
      VLOG(3) << "VMM V2 compact could not place remapped backing: pool="
              << static_cast<int>(pool_type_) << " requested=" << requested_size
              << " candidates=" << compact_result.remapped_handle_count
              << " event_blocked=" << stats.event_blocked_count
              << " partial=" << stats.partial_count
              << " remap_owned=" << stats.remapped_blocked_count
              << " unknown_safety=" << stats.unknown_safety_blocked_count;
      return 0;
    }
    return compact_result.remapped_bytes;
  } catch (...) {
    transaction.Rollback();
    throw;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
