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

#include <exception>
#include <utility>
#include <vector>

#include "glog/logging.h"

namespace paddle {
namespace memory {
namespace allocation {

size_t FreeBlockRemapCompactor::Compact(std::list<BlockV2>* blocks,
                                        size_t requested_size,
                                        uint64_t compact_seq) {
  const size_t handle_size = vmm_allocator_->handle_size();
  RemapTransaction transaction(vmm_allocator_.get(),
                               handle_size,
                               commit_synthetic_allocation_,
                               can_use_destination_range_,
                               release_stale_destination_allocations_);

  // Clear any sticky CUDA error before we start.
  cudaGetLastError();
  // No cudaDeviceSynchronize here. Source collection uses per-event query
  // inside RemapTransaction to avoid a full pipeline stall.

  // The entire compact is wrapped in try-catch.  If ANY CUDA API call
  // fails (e.g. cuMemUnmap during Phase 1, cuMemMap during Phase 2),
  // we rollback all unmapped handles via unmapped-free scatter so that the
  // block
  // list stays consistent for subsequent FreeImpl/TryMerge calls.
  try {
    // ---- Phase 1: Unmap fully-covered handles from FREE blocks ----
    if (VLOG_IS_ON(4)) {
      auto prescan = transaction.PreparePhase1Diagnostics(
          blocks,
          requested_size,
          "FreeBlockRemapCompactor::pre_phase1",
          "FreeBlockRemapCompactor::pre_phase1_target");
      VLOG(4) << "VMM V2 compactor BackingMap pre-scan pool="
              << static_cast<int>(pool_type_)
              << " free_ranges=" << prescan.free_range_count
              << " unmapped_free_ranges=" << prescan.unmapped_free_range_count
              << " mapped_pages=" << prescan.mapped_page_count
              << " mapped_bytes=" << prescan.mapped_page_count * handle_size
              << " target_unmapped_pages=" << prescan.target_page_count
              << " target_unmapped_bytes="
              << prescan.target_page_count * handle_size
              << " requested=" << requested_size
              << " snapshot_ok=" << prescan.source_ok
              << " target_snapshot_ok=" << prescan.target_ok;
    }
    auto compact_result =
        transaction.CompactFreeBlocks(blocks, requested_size, pool_type_);
    const auto& stats = compact_result.source_stats;

    // Keep per-compact coverage diagnostics behind VLOG; dsv3 regression can
    // execute many compactions and default INFO logging dominates runtime.
    VLOG(4) << "VMM V2 compactor pool=" << static_cast<int>(pool_type_)
            << " Phase 1 stats: free_blocks=" << stats.free_block_count
            << " safe_blocks=" << stats.safe_block_count
            << " event_blocked=" << stats.event_blocked_count
            << " event_blocked_bytes=" << stats.event_blocked_bytes
            << " fully_covered_parts=" << stats.fully_covered_count
            << " fully_covered_bytes=" << stats.fully_covered_bytes
            << " partial_parts=" << stats.partial_count
            << " partial_bytes=" << stats.partial_bytes
            << " unknown_safety_blocked=" << stats.unknown_safety_blocked_count
            << " unknown_safety_blocked_bytes="
            << stats.unknown_safety_blocked_bytes
            << " remapped_blocked=" << stats.remapped_blocked_count
            << " remapped_blocked_bytes=" << stats.remapped_blocked_bytes;
    if (!compact_result.success) {
      VLOG(3) << "VMM V2 compact summary: seq=" << compact_seq
              << " pool=" << static_cast<int>(pool_type_)
              << " requested=" << requested_size
              << " success=" << compact_result.success
              << " remapped_handles=" << compact_result.remapped_handle_count
              << " remapped_bytes=" << compact_result.remapped_bytes
              << " used_tail=" << compact_result.used_tail
              << " source_collect_us=" << compact_result.source_collect_us
              << " destination_plan_us=" << compact_result.destination_plan_us
              << " move_commit_us=" << compact_result.move_commit_us
              << " free_blocks=" << stats.free_block_count
              << " safe_blocks=" << stats.safe_block_count
              << " fully_covered=" << stats.fully_covered_count
              << " event_blocked=" << stats.event_blocked_count
              << " unknown_safety_blocked="
              << stats.unknown_safety_blocked_count
              << " remapped_blocked=" << stats.remapped_blocked_count;
    } else {
      VLOG(3) << "VMM V2 compact summary: seq=" << compact_seq
              << " pool=" << static_cast<int>(pool_type_)
              << " requested=" << requested_size
              << " remapped_handles=" << compact_result.remapped_handle_count
              << " remapped_bytes=" << compact_result.remapped_bytes
              << " move_commit_us=" << compact_result.move_commit_us;
    }
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

    VLOG(10) << "VMM remap compact pool=" << static_cast<int>(pool_type_)
             << " remapped_handles=" << compact_result.remapped_handle_count
             << " total_remapped=" << compact_result.remapped_bytes
             << " handle_size=" << handle_size;

    if (!compact_result.success) {
      return 0;
    }
    return compact_result.remapped_bytes;
  } catch (const std::exception& e) {
    VLOG(3) << "VMM V2 compactor: exception caught during Compact: " << e.what()
            << "; rolling back transaction";
    transaction.Rollback();
    return 0;
  } catch (...) {
    auto eptr = std::current_exception();
    VLOG(3) << "VMM V2 compactor: non-std exception caught during Compact, "
               "rolling back transaction. exception_ptr="
            << (eptr ? "set" : "null");
    transaction.Rollback();
    return 0;
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
