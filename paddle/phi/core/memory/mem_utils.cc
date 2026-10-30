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
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <system_error>

#include "glog/logging.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/memory/malloc.h"
#include "paddle/phi/core/memory/mem_visitor.h"
#include "paddle/phi/core/memory/stats.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/scope_guard.h"
#include "paddle/utils/string/printf.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"
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
  return std::move(all_blocks_info_visitor).GetAllBlocksInfo();
}

std::vector<std::vector<std::tuple<size_t, uintptr_t, bool>>>
LargePoolBlockInfo(const GPUPlace& place) {
  AllBlocksInfoVisitor visitor(AllBlocksInfoVisitor::PoolFilter::kLargeOnly);
  allocation::AllocatorFacade::Instance().Accept(place, &visitor);
  return std::move(visitor).GetAllBlocksInfo();
}

std::vector<std::vector<std::tuple<size_t, uintptr_t, bool>>>
SmallPoolBlockInfo(const GPUPlace& place) {
  AllBlocksInfoVisitor visitor(AllBlocksInfoVisitor::PoolFilter::kSmallOnly);
  allocation::AllocatorFacade::Instance().Accept(place, &visitor);
  return std::move(visitor).GetAllBlocksInfo();
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

std::vector<MemHistoryTraceEntry> GpuMemoryHistoryTrace(const GPUPlace& place) {
  return MemoryHistoryRecorder::Instance().GetTrace(place.GetDeviceId());
}

namespace {

using BlockInfoGroups =
    std::vector<std::vector<std::tuple<size_t, uintptr_t, bool>>>;

// mem_info.core.<local YYYYmmdd_HHMMSS>_<usec>_gpu<device>_pid<pid>_<seq>.
// pid separates ranks sharing a working directory; seq separates dumps of one
// process within the same microsecond.
std::string MakeBlockInfoDumpFileName(int device) {
  static std::atomic<uint64_t> sequence{0};
#ifdef _WIN32
  const auto pid = _getpid();
#else
  const auto pid = getpid();
#endif
  const auto now = std::chrono::system_clock::now();
  const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                          now.time_since_epoch())
                          .count() %
                      1000000;
  std::tm local_time{};
#ifdef _WIN32
  localtime_s(&local_time, &seconds);
#else
  localtime_r(&seconds, &local_time);
#endif
  // The 15-character stamp always fits; the buffer stays empty otherwise.
  char stamp[32] = {};
  std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &local_time);
  const int usec = static_cast<int>(micros);
  const int process_id = static_cast<int>(pid);
  const uint64_t seq = sequence.fetch_add(1, std::memory_order_relaxed);
  return string::Sprintf("mem_info.core.%s_%06d_gpu%d_pid%d_%d",
                         stamp,
                         usec,
                         device,
                         process_id,
                         seq);
}

// Same format as memory_analyzer.format_size.
std::string FormatMemorySize(size_t size) {
  if (size == 0) {
    return "0 B";
  }
  if (size < (1ULL << 20)) {
    return string::Sprintf("%.2f KB", static_cast<double>(size) / (1ULL << 10));
  }
  if (size < (1ULL << 30)) {
    return string::Sprintf("%.2f MB", static_cast<double>(size) / (1ULL << 20));
  }
  return string::Sprintf("%.2f GB", static_cast<double>(size) / (1ULL << 30));
}

// Append one memory_summary style table with left-aligned cells.
void AppendTable(std::ostream& out,
                 const std::string& title,
                 const std::vector<std::string>& headers,
                 const std::vector<std::vector<std::string>>& rows) {
  std::vector<size_t> widths;
  for (const auto& header : headers) {
    widths.push_back(header.size());
  }
  for (const auto& row : rows) {
    for (size_t i = 0; i < row.size(); ++i) {
      widths[i] = std::max(widths[i], row[i].size());
    }
  }
  const auto separator = [&] {
    out << '+';
    for (size_t width : widths) {
      out << std::string(width + 2, '-') << '+';
    }
    out << '\n';
  };
  const auto append_row = [&](const std::vector<std::string>& row) {
    out << '|';
    for (size_t i = 0; i < row.size(); ++i) {
      out << ' ' << row[i] << std::string(widths[i] - row[i].size() + 1, ' ')
          << '|';
    }
    out << '\n';
  };
  out << "\n### " << title << '\n';
  separator();
  append_row(headers);
  separator();
  for (const auto& row : rows) {
    append_row(row);
  }
  separator();
}

}  // namespace

std::string MessageSectionHeader(const std::string& title) {
  // Keep the 22-dash width of "Error Message Summary:" for short titles.
  const std::string line(std::max<size_t>(title.size(), 22), '-');
  return "\n" + line + "\n" + title + "\n" + line + "\n";
}

std::string DumpAllBlockInfoToFile(const BlockInfoGroups& blocks, int device) {
  const auto file_name = MakeBlockInfoDumpFileName(device);
  std::error_code path_error;
  const auto absolute_path = std::filesystem::absolute(file_name, path_error);
  const std::string display_path =
      path_error ? file_name : absolute_path.string();
  const auto failure = [&display_path](int error) {
    return "Failed to write all_block_info to " + display_path + ": " +
           std::error_code(error, std::generic_category()).message() + ".\n";
  };
  // Exclusive creation never clobbers an existing file.
  const auto close_file = [](FILE* file) { std::fclose(file); };
  std::unique_ptr<FILE, decltype(close_file)> file(
      std::fopen(file_name.c_str(), "wx"), close_file);
  if (!file) {
    return failure(errno);
  }
  bool committed = false;
  DEFINE_PADDLE_SCOPE_GUARD([&] {
    if (!committed) {
      // Do not leave a partially written dump behind.
      file.reset();
      std::remove(file_name.c_str());
    }
  });
  std::fputs("[\n", file.get());
  for (const auto& group : blocks) {
    std::fputs("  [\n", file.get());
    for (const auto& [size, address, is_free] : group) {
      std::fprintf(file.get(),
                   "    (%zu, %llu, %s),\n",
                   size,
                   static_cast<unsigned long long>(address),  // NOLINT
                   is_free ? "True" : "False");
    }
    std::fputs("  ],\n", file.get());
  }
  std::fputs("]\n", file.get());
  if (std::ferror(file.get())) {
    return failure(errno);
  }
  if (std::fclose(file.release()) != 0) {
    return failure(errno);
  }
  committed = true;
  return "all_block_info saved to " + display_path + "\n";
}

std::string AllBlockInfoSummary(const BlockInfoGroups& blocks,
                                const std::vector<std::string>& group_names) {
  size_t block_count = 0;
  size_t mapped = 0;
  size_t free_size = 0;
  size_t free_blocks = 0;
  size_t max_free = 0;
  for (const auto& group : blocks) {
    block_count += group.size();
    for (const auto& [size, address, is_free] : group) {
      mapped += size;
      if (is_free) {
        ++free_blocks;
        free_size += size;
        max_free = std::max(max_free, size);
      }
    }
  }
  std::string names;
  for (const auto& name : group_names) {
    names += (names.empty() ? "" : ", ") + name;
  }
  return string::Sprintf(
      "all_block_info summary: groups=%d (%s), blocks=%d, mapped=%s, "
      "used=%s, free=%s, free_blocks=%d, max_free=%s\n",
      blocks.size(),
      names.empty() ? "no mapped pool" : names,
      block_count,
      FormatMemorySize(mapped),
      FormatMemorySize(mapped - free_size),
      FormatMemorySize(free_size),
      free_blocks,
      FormatMemorySize(max_free));
}

std::string MemorySummaryTables(const BlockInfoGroups& blocks,
                                const GPUPlace& place,
                                int64_t allocated,
                                int64_t reserved,
                                size_t driver_available,
                                size_t driver_total) {
  std::ostringstream out;
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.imbue(std::locale::classic());
  const auto max_allocated =
      DeviceMemoryStatPeakValue("Allocated", place.device);
  const auto max_reserved = DeviceMemoryStatPeakValue("Reserved", place.device);
  // Same formula as memory_analyzer.py's "Max Frag Rate".
  const double max_frag_rate =
      reserved > 0 ? (static_cast<double>(reserved) - max_allocated) / reserved
                   : 0.0;
  std::vector<std::string> global_row;
  global_row.push_back(std::to_string(blocks.size()));
  global_row.push_back(FormatMemorySize(std::max<int64_t>(allocated, 0)));
  global_row.push_back(FormatMemorySize(std::max<int64_t>(max_allocated, 0)));
  global_row.push_back(FormatMemorySize(std::max<int64_t>(reserved, 0)));
  global_row.push_back(FormatMemorySize(std::max<int64_t>(max_reserved, 0)));
  global_row.push_back(string::Sprintf("%.2f%%", max_frag_rate * 100));
  global_row.push_back(FormatMemorySize(driver_total - driver_available) +
                       " / " + FormatMemorySize(driver_total));
  AppendTable(out,
              "Global Memory Snapshot",
              {"Allocators",
               "Allocated",
               "Max Alloc",
               "Reserved",
               "Max Reserved",
               "Max Frag Rate",
               "Phy GPU Used / Total"},
              {global_row});

  // Keep the bucket boundaries identical to memory_analyzer.py; a block of
  // exactly a threshold size belongs to the next bucket.
  constexpr std::array<size_t, 11> thresholds = {1ULL << 20,
                                                 10ULL << 20,
                                                 50ULL << 20,
                                                 100ULL << 20,
                                                 200ULL << 20,
                                                 400ULL << 20,
                                                 600ULL << 20,
                                                 800ULL << 20,
                                                 1ULL << 30,
                                                 2ULL << 30,
                                                 3ULL << 30};
  std::vector<std::vector<std::string>> summary_rows;
  std::vector<std::vector<std::string>> distribution_rows;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto name = "Allocator_" + std::to_string(i);
    size_t total_size = 0;
    size_t free_size = 0;
    size_t free_blocks = 0;
    size_t max_used = 0;
    size_t max_free = 0;
    std::array<std::array<size_t, 2>, thresholds.size() + 1> buckets{};
    for (const auto& [size, address, is_free] : blocks[i]) {
      total_size += size;
      if (is_free) {
        ++free_blocks;
        free_size += size;
        max_free = std::max(max_free, size);
      } else {
        max_used = std::max(max_used, size);
      }
      const auto bucket =
          std::upper_bound(thresholds.begin(), thresholds.end(), size) -
          thresholds.begin();
      ++buckets[bucket][is_free ? 0 : 1];
    }
    summary_rows.push_back({name,
                            std::to_string(blocks[i].size()),
                            std::to_string(blocks[i].size() - free_blocks),
                            std::to_string(free_blocks),
                            FormatMemorySize(total_size),
                            FormatMemorySize(total_size - free_size),
                            FormatMemorySize(free_size),
                            FormatMemorySize(max_used),
                            FormatMemorySize(max_free)});
    std::vector<std::string> free_row{name, "Free Blocks"};
    std::vector<std::string> used_row{name, "Used Blocks"};
    for (const auto& bucket : buckets) {
      free_row.push_back(std::to_string(bucket[0]));
      used_row.push_back(std::to_string(bucket[1]));
    }
    distribution_rows.push_back(std::move(free_row));
    distribution_rows.push_back(std::move(used_row));
  }
  AppendTable(out,
              "Allocator Summary Statistics",
              {"ID",
               "Tot Blks",
               "Used Blks",
               "Free Blks",
               "Tot Size",
               "Used Size",
               "Free Size",
               "Max Used",
               "Max Free"},
              summary_rows);
  AppendTable(out,
              "Block Size Distribution",
              {"Allocator ID",
               "Block Type",
               "[0B,1M)",
               "[1M,10M)",
               "[10M,50M)",
               "[50M,100M)",
               "[100M,200M)",
               "[200M,400M)",
               "[400M,600M)",
               "[600M,800M)",
               "[800M,1G)",
               "[1G,2G)",
               "[2G,3G)",
               "[3G,+INF)"},
              distribution_rows);
  return out.str();
}

std::string BuildOOMMemoryDiagnostics(
    allocation::VMMAutoGrowthBestFitMultiPoolAllocatorV2* allocator,
    const GPUPlace& place,
    int64_t allocated,
    int64_t reserved,
    size_t driver_available,
    size_t driver_total) {
  try {
    BlockInfoGroups blocks;
    std::vector<std::string> group_names;
    // Match all_block_info's pool grouping, but inspect the failing allocator
    // directly rather than the default compute-stream allocator.
    const std::pair<
        const char*,
        std::shared_ptr<allocation::VMMAutoGrowthBestFitAllocatorV2>>
        pools[] = {{"small", allocator->small_allocator()},
                   {"large", allocator->large_allocator()}};
    for (const auto& [pool_name, pool] : pools) {
      if (pool) {
        auto snapshot = pool->SnapshotBlockInfo();
        if (!snapshot.empty()) {
          group_names.push_back(
              string::Sprintf("group %d=%s pool", blocks.size(), pool_name));
          blocks.push_back(std::move(snapshot));
        }
      }
    }
    const std::string dump_result =
        DumpAllBlockInfoToFile(blocks, place.device);
    LOG(WARNING) << "VMM V2 final OOM on GPU " << static_cast<int>(place.device)
                 << ": " << dump_result;
    return MessageSectionHeader(
               string::Sprintf("Memory Status SnapShot (GPU %d)",
                               static_cast<int>(place.device))) +
           string::Sprintf(
               "Block statistics cover the failing allocator's nonempty pools "
               "(small, then large); device statistics cover all "
               "allocators.\n"
               "%s%s%s",
               AllBlockInfoSummary(blocks, group_names),
               dump_result,
               MemorySummaryTables(blocks,
                                   place,
                                   allocated,
                                   reserved,
                                   driver_available,
                                   driver_total));
  } catch (...) {
    // Diagnostics must not replace the allocation failure, e.g. when device
    // statistics are unavailable or the host cannot build the tables.
    return "all_block_info / memory_summary diagnostics unavailable.\n";
  }
}

#endif

}  // namespace memory
}  // namespace paddle
