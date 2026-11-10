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

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "paddle/common/flags.h"
#include "paddle/phi/core/memory/allocation/cuda_allocator.h"
#include "paddle/phi/core/memory/allocation/retry_allocator.h"
#include "paddle/phi/core/memory/allocation/stat_allocator.h"
#include "paddle/phi/core/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/mem_utils.h"
#include "paddle/phi/core/scope_guard.h"

#ifndef _WIN32
#include <signal.h>
#include <sys/resource.h>
#endif

COMMON_DECLARE_int64(offload_retry_times);
COMMON_DECLARE_bool(vmm_v2_remap_on_oom);

namespace paddle {
namespace memory {
namespace allocation {

namespace {

__global__ void VMMStreamSafeBusyWait(uint64_t cycles) {
  const auto start = clock64();
  while (clock64() - start < cycles) {
  }
}

std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> CreatePoolAllocator(
    size_t handle_size, PoolType pool_type) {
  auto underlying = std::make_shared<CUDAVirtualMemAllocatorV2>(
      phi::GPUPlace(), handle_size, pool_type);
  return std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
      underlying, 256, phi::GPUPlace(), pool_type);
}

std::unique_ptr<VMMAutoGrowthBestFitMultiPoolAllocatorV2> CreateAllocator() {
  return std::make_unique<VMMAutoGrowthBestFitMultiPoolAllocatorV2>(
      CreatePoolAllocator(2UL << 20, PoolType::kSmall),
      CreatePoolAllocator(2UL << 20, PoolType::kLarge),
      2UL << 20,
      phi::GPUPlace());
}

class FailingMultiPoolAllocator
    : public VMMAutoGrowthBestFitMultiPoolAllocatorV2 {
 public:
  explicit FailingMultiPoolAllocator(size_t requested_handles = 2,
                                     size_t created_handles = 1,
                                     size_t success_on_attempt = 0,
                                     size_t small_allocation_threshold = 2UL
                                                                         << 20)
      : FailingMultiPoolAllocator(
            CreatePoolAllocator(2UL << 20, PoolType::kSmall),
            CreatePoolAllocator(2UL << 20, PoolType::kLarge),
            requested_handles,
            created_handles,
            success_on_attempt,
            small_allocation_threshold) {}

  FailingMultiPoolAllocator(
      const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>& small,
      const std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2>& large,
      size_t requested_handles = 2,
      size_t created_handles = 1,
      size_t success_on_attempt = 0,
      size_t small_allocation_threshold = 2UL << 20)
      : VMMAutoGrowthBestFitMultiPoolAllocatorV2(
            small, large, small_allocation_threshold, phi::GPUPlace()),
        requested_handles_(requested_handles),
        created_handles_(created_handles),
        success_on_attempt_(success_on_attempt),
        small_allocation_threshold_(small_allocation_threshold) {}

  size_t allocation_attempts() const { return allocation_attempts_; }
  void SetAllocationObserver(std::function<void(size_t)> observer) {
    allocation_observer_ = std::move(observer);
  }

 protected:
  phi::Allocation* AllocateImpl(size_t size) override {
    ++allocation_attempts_;
    if (allocation_observer_) {
      allocation_observer_(allocation_attempts_);
    }
    if (allocation_attempts_ == success_on_attempt_) {
      return VMMAutoGrowthBestFitMultiPoolAllocatorV2::AllocateImpl(size);
    }
    const auto pool_type = AlignedSize(size, small_allocator()->alignment()) <
                                   small_allocation_threshold_
                               ? PoolType::kSmall
                               : PoolType::kLarge;
    throw VMMGrowOOM("deterministic VMM stream-safe allocation failure " +
                         std::to_string(allocation_attempts_),
                     __FILE__,
                     __LINE__,
                     VMMGrowOOMInfo{/*requested_handles=*/requested_handles_,
                                    /*created_handles=*/created_handles_,
                                    /*handle_size=*/2UL << 20,
                                    /*device=*/0,
                                    pool_type});
  }

 private:
  size_t allocation_attempts_{0};
  size_t requested_handles_;
  size_t created_handles_;
  size_t success_on_attempt_;
  size_t small_allocation_threshold_;
  std::function<void(size_t)> allocation_observer_;
};

class FailingUnmapAllocator : public CUDAVirtualMemAllocatorV2 {
 public:
  FailingUnmapAllocator()
      : CUDAVirtualMemAllocatorV2(
            phi::GPUPlace(), 2UL << 20, PoolType::kLarge) {}

  void FailNextUnmap() { fail_next_unmap_ = true; }
  size_t unmap_calls() const { return unmap_calls_; }

 protected:
  CUresult UnmapRangeForRelease(VMMDevicePtr ptr, size_t size) override {
    ++unmap_calls_;
    if (fail_next_unmap_) {
      fail_next_unmap_ = false;
      return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDAVirtualMemAllocatorV2::UnmapRangeForRelease(ptr, size);
  }

 private:
  bool fail_next_unmap_{false};
  size_t unmap_calls_{0};
};

class ReleaseCountingStreamSafeCUDAAllocator : public StreamSafeCUDAAllocator {
 public:
  using StreamSafeCUDAAllocator::StreamSafeCUDAAllocator;

  size_t release_calls() const { return release_calls_; }
  uint64_t released_bytes() const { return released_bytes_; }
  void ReportZeroReleasedBytes() { report_zero_released_bytes_ = true; }
  void SetBeforeReleaseObserver(std::function<void()> observer) {
    before_release_observer_ = std::move(observer);
  }
  void SetReleaseObserver(std::function<void()> observer) {
    release_observer_ = std::move(observer);
  }

 protected:
  uint64_t ReleaseImpl(const Place& place) override {
    ++release_calls_;
    if (before_release_observer_) {
      before_release_observer_();
    }
    released_bytes_ = StreamSafeCUDAAllocator::ReleaseImpl(place);
    if (release_observer_) {
      release_observer_();
    }
    // Exercise the zero-return path while retaining real coordinated release.
    return report_zero_released_bytes_ ? 0 : released_bytes_;
  }

 private:
  size_t release_calls_{0};
  uint64_t released_bytes_{0};
  bool report_zero_released_bytes_{false};
  std::function<void()> before_release_observer_;
  std::function<void()> release_observer_;
};

class RetryWithoutRemapMultiPoolAllocator
    : public VMMAutoGrowthBestFitMultiPoolAllocatorV2 {
 public:
  explicit RetryWithoutRemapMultiPoolAllocator(size_t success_on_attempt)
      : VMMAutoGrowthBestFitMultiPoolAllocatorV2(
            CreatePoolAllocator(2UL << 20, PoolType::kSmall),
            CreatePoolAllocator(2UL << 20, PoolType::kLarge),
            2UL << 20,
            phi::GPUPlace()),
        success_on_attempt_(success_on_attempt) {}

  void PrepareRetryBlock(size_t size) {
    retry_block_ = large_allocator()->Allocate(size);
  }

  size_t allocation_attempts() const { return allocation_attempts_; }

 protected:
  phi::Allocation* AllocateImpl(size_t size) override {
    ++allocation_attempts_;
    if (allocation_attempts_ == 2) {
      // Simulate another thread freeing a reusable block after the second
      // allocation failure and before OOM remap precheck.
      retry_block_.reset();
    }
    if (allocation_attempts_ != success_on_attempt_) {
      throw VMMGrowOOM("deterministic concurrent-free allocation failure " +
                           std::to_string(allocation_attempts_),
                       __FILE__,
                       __LINE__,
                       VMMGrowOOMInfo{/*requested_handles=*/2,
                                      /*created_handles=*/0,
                                      /*handle_size=*/2UL << 20,
                                      /*device=*/0,
                                      PoolType::kLarge});
    }
    return VMMAutoGrowthBestFitMultiPoolAllocatorV2::AllocateImpl(size);
  }

 private:
  AllocationPtr retry_block_;
  size_t allocation_attempts_{0};
  size_t success_on_attempt_;
};

class ScopedVLogLevel {
 public:
  explicit ScopedVLogLevel(int level) : old_level_(FLAGS_v) { FLAGS_v = level; }
  ~ScopedVLogLevel() { FLAGS_v = old_level_; }

 private:
  int old_level_;
};

void PrepareRemapSource(VMMAutoGrowthBestFitMultiPoolAllocatorV2* allocator,
                        AllocationPtr* anchor,
                        PoolType pool_type = PoolType::kLarge) {
  const auto& pool = pool_type == PoolType::kSmall
                         ? allocator->small_allocator()
                         : allocator->large_allocator();
  auto source = pool->Allocate(2UL << 20);
  *anchor = pool->Allocate(2UL << 20);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(*anchor, nullptr);
  auto* remap_source = dynamic_cast<VMMRemapEventAllocation*>(source.get());
  ASSERT_NE(remap_source, nullptr);
  ASSERT_TRUE(remap_source->SetVMMRemapEvent(cudaStreamPerThread, nullptr));
  source.reset();
}

bool HasBlockWithPtr(const VMMAutoGrowthBestFitAllocatorV2& allocator,
                     void* ptr,
                     BlockType type) {
  for (const auto& block : allocator.all_blocks()) {
    if (block.ptr() == ptr && block.type_ == type) {
      return true;
    }
  }
  return false;
}

class CountingV2AllocatorVisitor : public paddle::memory::AllocatorVisitor {
 public:
  using paddle::memory::AllocatorVisitor::Visit;

  void Visit(VMMAutoGrowthBestFitAllocatorV2* allocator) override {
    ASSERT_NE(allocator, nullptr);
    ++v2_pool_visits_;
  }

  int v2_pool_visits() const { return v2_pool_visits_; }

 private:
  int v2_pool_visits_{0};
};

class ScopedVMMRetryFlags {
 public:
  ScopedVMMRetryFlags(int64_t retry_times, bool remap_on_oom)
      : retry_times_(FLAGS_offload_retry_times),
        remap_on_oom_(FLAGS_vmm_v2_remap_on_oom) {
    FLAGS_offload_retry_times = retry_times;
    FLAGS_vmm_v2_remap_on_oom = remap_on_oom;
  }

  ~ScopedVMMRetryFlags() {
    RegisterOOMCallback(nullptr);
    FLAGS_offload_retry_times = retry_times_;
    FLAGS_vmm_v2_remap_on_oom = remap_on_oom_;
  }

 private:
  int64_t retry_times_;
  bool remap_on_oom_;
};

class ScopedTemporaryCwd {
 public:
  ScopedTemporaryCwd() : old_path_(std::filesystem::current_path()) {
    static std::atomic<uint64_t> next_id{0};
    const auto base = std::filesystem::temp_directory_path();
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (size_t attempt = 0; attempt < 100; ++attempt) {
      temp_path_ = base / ("paddle_vmm_v2_test_" + std::to_string(stamp) + "_" +
                           std::to_string(next_id.fetch_add(1)));
      if (std::filesystem::create_directory(temp_path_)) {
        break;
      }
      temp_path_.clear();
    }
    if (temp_path_.empty()) {
      throw std::runtime_error("failed to create temporary VMM test directory");
    }
    std::error_code error;
    std::filesystem::current_path(temp_path_, error);
    if (error) {
      std::error_code cleanup_error;
      std::filesystem::remove_all(temp_path_, cleanup_error);
      throw std::system_error(error, "failed to switch to VMM test directory");
    }
  }

  ~ScopedTemporaryCwd() {
    std::error_code error;
    std::filesystem::current_path(old_path_, error);
    EXPECT_FALSE(error) << error.message();
    std::filesystem::remove_all(temp_path_, error);
    EXPECT_FALSE(error) << error.message();
  }

  ScopedTemporaryCwd(const ScopedTemporaryCwd&) = delete;
  ScopedTemporaryCwd& operator=(const ScopedTemporaryCwd&) = delete;

 private:
  std::filesystem::path old_path_;
  std::filesystem::path temp_path_;
};

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  EXPECT_TRUE(input.is_open()) << path;
  std::ostringstream contents;
  contents << input.rdbuf();
  EXPECT_FALSE(input.bad()) << path;
  return contents.str();
}

std::string RemoveWhitespace(const std::string& text) {
  std::string compact;
  compact.reserve(text.size());
  for (const unsigned char ch : text) {
    if (!std::isspace(ch)) {
      compact.push_back(static_cast<char>(ch));
    }
  }
  return compact;
}

std::string BlockInfoTuple(
    const VMMAutoGrowthBestFitAllocatorV2::BlockInfo& block) {
  return "(" + std::to_string(std::get<0>(block)) + "," +
         std::to_string(std::get<1>(block)) + "," +
         (std::get<2>(block) ? "True" : "False") + ")";
}

// Returns the dump path printed by the final OOM message ("" if absent).
std::filesystem::path SavedDumpPath(const std::string& message) {
  constexpr char kMarker[] = "all_block_info saved to ";
  const auto start = message.find(kMarker);
  if (start == std::string::npos) {
    return {};
  }
  const auto path_start = start + sizeof(kMarker) - 1;
  return message.substr(path_start,
                        message.find('\n', path_start) - path_start);
}

// mem_info.core.* files in the current directory.
std::vector<std::filesystem::path> DumpFiles() {
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(".")) {
    if (entry.path().filename().string().rfind("mem_info.core", 0) == 0) {
      files.push_back(entry.path().filename());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

// Checks that the message names a new absolute dump path in the current
// directory and that the file holds exactly `pools`.
void ExpectBlockInfoDump(
    const std::string& message,
    const std::vector<std::vector<VMMAutoGrowthBestFitAllocatorV2::BlockInfo>>&
        pools) {
  const auto path = SavedDumpPath(message);
  ASSERT_FALSE(path.empty()) << message;
  EXPECT_TRUE(path.is_absolute()) << path;
  EXPECT_EQ(path.parent_path(), std::filesystem::current_path());
  const auto name = path.filename().string();
  EXPECT_EQ(name.rfind("mem_info.core.", 0), 0UL) << name;
  EXPECT_NE(name.find("_gpu0_pid"), std::string::npos) << name;
  std::string expected = "[";
  for (const auto& blocks : pools) {
    if (blocks.empty()) {
      continue;
    }
    if (expected.back() != '[') {
      expected += ',';
    }
    expected += '[';
    for (const auto& block : blocks) {
      if (expected.back() != '[') {
        expected += ',';
      }
      expected += BlockInfoTuple(block);
    }
    expected += ']';
  }
  expected += ']';
  // Ignore whitespace and optional trailing commas, retaining pool boundaries,
  // tuple order, decimal addresses and Python boolean spelling.
  std::string actual;
  for (const char ch : RemoveWhitespace(ReadTextFile(path))) {
    if ((ch == ']' || ch == ')') && !actual.empty() && actual.back() == ',') {
      actual.pop_back();
    }
    actual += ch;
  }
  EXPECT_EQ(actual, expected);
}

void ExpectMemorySummary(const std::string& message) {
  // Section titles are framed by dash lines exactly as long as the title.
  for (const char* title :
       {"\n------------------------------\nMemory Status SnapShot "
        "(GPU 0)\n------------------------------\n",
        "\n---------------------------\nAllocation Failure "
        "Summary:\n---------------------------\n",
        "Global Memory Snapshot",
        "Allocator Summary Statistics",
        "Block Size Distribution"}) {
    EXPECT_NE(message.find(title), std::string::npos) << title;
  }
  const auto compact = RemoveWhitespace(message);
  for (const char* header :
       {"|Allocators|Allocated|MaxAlloc|Reserved|MaxReserved|MaxFragRate|"
        "PhyGPUUsed/Total|",
        "|ID|TotBlks|UsedBlks|FreeBlks|TotSize|UsedSize|FreeSize|MaxUsed|"
        "MaxFree|",
        "|AllocatorID|BlockType|[0B,1M)|[1M,10M)|[10M,50M)|[50M,100M)|"
        "[100M,200M)|[200M,400M)|[400M,600M)|[600M,800M)|[800M,1G)|"
        "[1G,2G)|[2G,3G)|[3G,+INF)|"}) {
    EXPECT_NE(compact.find(header), std::string::npos) << header;
  }
}

}  // namespace

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     AllocatorVisitorTraversesV2Pools) {
  auto allocator = CreateAllocator();
  EXPECT_TRUE(allocator->IsAllocThreadSafe());

  paddle::memory::AllocatorVisitor base_visitor;
  EXPECT_NO_THROW(allocator->small_allocator()->Accept(&base_visitor));

  CountingV2AllocatorVisitor counting_visitor;
  allocator->Accept(&counting_visitor);
  EXPECT_EQ(counting_visitor.v2_pool_visits(), 2);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, RouteSmallAndLarge) {
  auto allocator = CreateAllocator();

  auto small = allocator->Allocate(256UL);
  auto aligned_to_threshold = allocator->Allocate((2UL << 20) - 1);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(aligned_to_threshold, nullptr);
  ASSERT_NE(large, nullptr);

  EXPECT_TRUE(HasBlockWithPtr(
      *allocator->small_allocator(), small->ptr(), BlockType::kActive));
  EXPECT_FALSE(HasBlockWithPtr(
      *allocator->large_allocator(), small->ptr(), BlockType::kActive));
  EXPECT_TRUE(HasBlockWithPtr(*allocator->large_allocator(),
                              aligned_to_threshold->ptr(),
                              BlockType::kActive));
  EXPECT_FALSE(HasBlockWithPtr(*allocator->small_allocator(),
                               aligned_to_threshold->ptr(),
                               BlockType::kActive));
  EXPECT_TRUE(HasBlockWithPtr(
      *allocator->large_allocator(), large->ptr(), BlockType::kActive));
  EXPECT_FALSE(HasBlockWithPtr(
      *allocator->small_allocator(), large->ptr(), BlockType::kActive));

  size_t total_free = 0;
  size_t max_free = 0;
  allocator->GetFreeBlockStats(&total_free, &max_free, 256UL);
  EXPECT_GT(total_free, 0UL);
  EXPECT_GT(max_free, 0UL);
  allocator->GetFreeBlockStats(&total_free, &max_free, 0);
  EXPECT_GT(total_free, 0UL);
  EXPECT_GT(max_free, 0UL);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, SetBlockRemapEventRoutesByPtr) {
  auto allocator = CreateAllocator();

  auto allocation = allocator->Allocate(256UL);
  ASSERT_NE(allocation, nullptr);
  auto* remap_allocation =
      dynamic_cast<VMMRemapEventAllocation*>(allocation.get());
  ASSERT_NE(remap_allocation, nullptr);
  EXPECT_TRUE(remap_allocation->SetVMMRemapEvent(cudaStreamPerThread, nullptr));

  gpuEvent_t event = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            cudaSuccess);
  auto guard = std::make_shared<CUDAEventGuard>(event);
  auto* ptr = allocation->ptr();
  ASSERT_TRUE(allocator->SetBlockRemapEvent(ptr, nullptr, guard));
  EXPECT_FALSE(allocator->SetBlockRemapEvent(
      reinterpret_cast<void*>(0x1), nullptr, nullptr));
  EXPECT_TRUE(
      HasBlockWithPtr(*allocator->small_allocator(), ptr, BlockType::kActive));

  allocation.reset();
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, CompactRoutesByRequestSize) {
  auto allocator = CreateAllocator();

  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  auto large_anchor = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);
  ASSERT_NE(large_anchor, nullptr);

  auto* small_remap = dynamic_cast<VMMRemapEventAllocation*>(small.get());
  auto* large_remap = dynamic_cast<VMMRemapEventAllocation*>(large.get());
  ASSERT_NE(small_remap, nullptr);
  ASSERT_NE(large_remap, nullptr);
  ASSERT_TRUE(small_remap->SetVMMRemapEvent(cudaStreamPerThread, nullptr));
  ASSERT_TRUE(large_remap->SetVMMRemapEvent(cudaStreamPerThread, nullptr));

  small.reset();
  large.reset();

  // Matching threshold and handle size skips the small pool's remap precheck.
  VMMRemapAttemptResult small_result;
  EXPECT_EQ(allocator->RemapForAllocation(
                phi::GPUPlace(), 256UL, nullptr, &small_result),
            0UL);
  EXPECT_EQ(small_result.status, VMMRemapAttemptStatus::kNotAttempted);
  EXPECT_EQ(allocator->small_allocator()->all_blocks().front().type_,
            BlockType::kFree);
  EXPECT_EQ(allocator->large_allocator()->all_blocks().front().type_,
            BlockType::kFree);

  EXPECT_EQ(allocator->RemapForAllocation(phi::GPUPlace(), 4UL << 20),
            2UL << 20);
  EXPECT_EQ(allocator->small_allocator()->all_blocks().front().type_,
            BlockType::kFree);
  EXPECT_EQ(allocator->large_allocator()->all_blocks().front().type_,
            BlockType::kUnmappedFree);

  auto second_allocator = CreateAllocator();
  auto second_small = second_allocator->Allocate(256UL);
  auto second_large = second_allocator->Allocate(2UL << 20);
  auto second_anchor = second_allocator->Allocate(2UL << 20);
  ASSERT_NE(second_small, nullptr);
  ASSERT_NE(second_large, nullptr);
  ASSERT_NE(second_anchor, nullptr);
  auto* second_small_remap =
      dynamic_cast<VMMRemapEventAllocation*>(second_small.get());
  auto* second_large_remap =
      dynamic_cast<VMMRemapEventAllocation*>(second_large.get());
  ASSERT_NE(second_small_remap, nullptr);
  ASSERT_NE(second_large_remap, nullptr);
  ASSERT_TRUE(
      second_small_remap->SetVMMRemapEvent(cudaStreamPerThread, nullptr));
  ASSERT_TRUE(
      second_large_remap->SetVMMRemapEvent(cudaStreamPerThread, nullptr));
  second_small.reset();
  second_large.reset();

  // With matching threshold and handle size, maintenance only compacts the
  // large pool.
  EXPECT_EQ(second_allocator->RemapForAllocation(phi::GPUPlace(), 0),
            2UL << 20);
  EXPECT_EQ(second_allocator->small_allocator()->all_blocks().front().type_,
            BlockType::kFree);
  EXPECT_EQ(second_allocator->large_allocator()->all_blocks().front().type_,
            BlockType::kUnmappedFree);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     SmallPoolRemapRequiresThresholdAboveHandleSize) {
  const phi::GPUPlace place(0);
  for (size_t handle_size : {2UL << 20, (2UL << 20) + 256UL, 4UL << 20}) {
    SCOPED_TRACE(handle_size);
    auto bottom = std::make_shared<CUDAVirtualMemAllocatorV2>(
        place, handle_size, PoolType::kSmall);
    auto small_pool = std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
        bottom, 256, place, PoolType::kSmall);
    EXPECT_EQ(small_pool->handle_size(), bottom->handle_size());
    auto allocation = small_pool->Allocate(256UL);
    allocation.reset();
    const size_t actual_handle_size = small_pool->handle_size();
    EXPECT_EQ(actual_handle_size, bottom->handle_size());
    EXPECT_GE(actual_handle_size, handle_size);
    for (size_t threshold :
         {actual_handle_size / 2, actual_handle_size, actual_handle_size * 2}) {
      SCOPED_TRACE(threshold);
      VMMAutoGrowthBestFitMultiPoolAllocatorV2 allocator(
          small_pool,
          CreatePoolAllocator(2UL << 20, PoolType::kLarge),
          threshold,
          place);
      // A reusable block distinguishes entering the pool's precheck from
      // returning immediately, and nondefault fields check result reset.
      VMMRemapAttemptResult result{VMMRemapAttemptStatus::kAttempted, 1, 1};
      EXPECT_EQ(allocator.RemapForAllocation(place, 256UL, nullptr, &result),
                0UL);
      if (threshold <= actual_handle_size) {
        EXPECT_EQ(result.status, VMMRemapAttemptStatus::kNotAttempted);
        EXPECT_EQ(result.movable_bytes, 0UL);
        EXPECT_EQ(result.required_bytes, 0UL);
      } else {
        EXPECT_EQ(result.status, VMMRemapAttemptStatus::kRetryWithoutRemap);
      }
      EXPECT_EQ(allocator.RemapForAllocation(place, 256UL), 0UL);
      EXPECT_EQ(small_pool->all_blocks().front().type_, BlockType::kFree);
      EXPECT_TRUE(allocator.large_allocator()->all_blocks().empty());
    }
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     MaintenanceRemapsSmallPoolWhenThresholdExceedsHandleSize) {
  const phi::GPUPlace place(0);
  constexpr size_t kHandleSize = 2UL << 20;
  for (size_t threshold : {1UL << 20, kHandleSize, 8UL << 20}) {
    SCOPED_TRACE(threshold);
    for (int source_pools = 0; source_pools < 4; ++source_pools) {
      SCOPED_TRACE(source_pools);
      auto allocator = std::make_shared<FailingMultiPoolAllocator>(
          /*requested_handles=*/2,
          /*created_handles=*/1,
          /*success_on_attempt=*/0,
          threshold);
      AllocationPtr small_anchor;
      AllocationPtr large_anchor;
      void* small_source = nullptr;
      void* large_source = nullptr;
      if (source_pools & 1) {
        PrepareRemapSource(allocator.get(), &small_anchor, PoolType::kSmall);
        small_source = allocator->small_allocator()->all_blocks().front().ptr();
      }
      if (source_pools & 2) {
        PrepareRemapSource(allocator.get(), &large_anchor, PoolType::kLarge);
        large_source = allocator->large_allocator()->all_blocks().front().ptr();
      }
      ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
      const bool remap_small =
          small_source != nullptr && threshold > kHandleSize;
      const size_t expected_bytes =
          (static_cast<size_t>(remap_small) + (large_source != nullptr)) *
          kHandleSize;
      VMMRemapAttemptResult result{VMMRemapAttemptStatus::kNotAttempted, 1, 1};
      EXPECT_EQ(allocator->RemapForAllocation(place, 0, nullptr, &result),
                expected_bytes);
      EXPECT_EQ(result.status,
                expected_bytes > 0 ? VMMRemapAttemptStatus::kAttempted
                                   : VMMRemapAttemptStatus::kNoMovableMemory);
      EXPECT_EQ(result.movable_bytes, expected_bytes);
      EXPECT_EQ(result.required_bytes, 0UL);
      if (small_source != nullptr) {
        EXPECT_TRUE(HasBlockWithPtr(
            *allocator->small_allocator(),
            small_source,
            remap_small ? BlockType::kUnmappedFree : BlockType::kFree));
        EXPECT_TRUE(HasBlockWithPtr(*allocator->small_allocator(),
                                    small_anchor->ptr(),
                                    BlockType::kActive));
      }
      if (large_source != nullptr) {
        EXPECT_TRUE(HasBlockWithPtr(*allocator->large_allocator(),
                                    large_source,
                                    BlockType::kUnmappedFree));
        EXPECT_TRUE(HasBlockWithPtr(*allocator->large_allocator(),
                                    large_anchor->ptr(),
                                    BlockType::kActive));
      }
      EXPECT_EQ(allocator->Compact(place), 0UL);
      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     MaintenanceWithoutResultRemapsBothPools) {
  const phi::GPUPlace place(0);
  for (int source_pools : {1, 2, 3}) {
    SCOPED_TRACE(source_pools);
    auto multi =
        std::make_shared<FailingMultiPoolAllocator>(2, 1, 0, 8UL << 20);
    AllocationPtr small_anchor;
    AllocationPtr large_anchor;
    void* small_source = nullptr;
    void* large_source = nullptr;
    if (source_pools & 1) {
      PrepareRemapSource(multi.get(), &small_anchor, PoolType::kSmall);
      small_source = multi->small_allocator()->all_blocks().front().ptr();
    }
    if (source_pools & 2) {
      PrepareRemapSource(multi.get(), &large_anchor, PoolType::kLarge);
      large_source = multi->large_allocator()->all_blocks().front().ptr();
    }
    ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
    EXPECT_EQ(multi->RemapForAllocation(place, 0),
              (source_pools == 3 ? 4UL : 2UL) << 20);
    if (small_source != nullptr) {
      EXPECT_TRUE(HasBlockWithPtr(
          *multi->small_allocator(), small_source, BlockType::kUnmappedFree));
      EXPECT_TRUE(HasBlockWithPtr(
          *multi->small_allocator(), small_anchor->ptr(), BlockType::kActive));
    }
    if (large_source != nullptr) {
      EXPECT_TRUE(HasBlockWithPtr(
          *multi->large_allocator(), large_source, BlockType::kUnmappedFree));
      EXPECT_TRUE(HasBlockWithPtr(
          *multi->large_allocator(), large_anchor->ptr(), BlockType::kActive));
    }
    EXPECT_EQ(multi->RemapForAllocation(place, 0), 0UL);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     MaintenancePreservesAttemptedStatusWithoutProgress) {
  const phi::GPUPlace place(0);
  constexpr size_t kHandleSize = 2UL << 20;
  for (PoolType blocked_pool : {PoolType::kSmall, PoolType::kLarge}) {
    SCOPED_TRACE(static_cast<int>(blocked_pool));
    for (bool other_pool_has_source : {false, true}) {
      SCOPED_TRACE(other_pool_has_source);
      auto bottom = std::make_shared<CUDAVirtualMemAllocatorV2>(
          place, kHandleSize, blocked_pool);
      auto blocked = std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
          bottom, 256, place, blocked_pool);
      auto small = blocked_pool == PoolType::kSmall
                       ? blocked
                       : CreatePoolAllocator(kHandleSize, PoolType::kSmall);
      auto large = blocked_pool == PoolType::kLarge
                       ? blocked
                       : CreatePoolAllocator(kHandleSize, PoolType::kLarge);
      VMMAutoGrowthBestFitMultiPoolAllocatorV2 multi(
          small, large, 8UL << 20, place);
      AllocationPtr blocked_anchor;
      AllocationPtr other_anchor;
      PrepareRemapSource(&multi, &blocked_anchor, blocked_pool);
      if (other_pool_has_source) {
        PrepareRemapSource(&multi,
                           &other_anchor,
                           blocked_pool == PoolType::kSmall ? PoolType::kLarge
                                                            : PoolType::kSmall);
      }
      // Block the tail outside the pool's block list, with no unmapped gaps
      // available. The source precheck succeeds but placement cannot proceed.
      auto hidden_tail = bottom->AppendWithBlock(kHandleSize);
      ASSERT_TRUE(hidden_tail.HasAllocation());
      ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
      const auto before = blocked->SnapshotBlockInfo();
      VMMRemapAttemptResult result;
      EXPECT_EQ(multi.RemapForAllocation(place, 0, nullptr, &result),
                other_pool_has_source ? kHandleSize : 0UL);
      EXPECT_EQ(result.status, VMMRemapAttemptStatus::kAttempted);
      EXPECT_EQ(result.movable_bytes,
                (other_pool_has_source ? 2UL : 1UL) * kHandleSize);
      EXPECT_EQ(result.required_bytes, 0UL);
      EXPECT_EQ(blocked->SnapshotBlockInfo(), before);
      // Exercise the same zero-progress path with no output result.
      EXPECT_EQ(multi.RemapForAllocation(place, 0), 0UL);
      EXPECT_EQ(blocked->SnapshotBlockInfo(), before);
    }
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     MaintenancePropagatesPoolExceptions) {
  const phi::GPUPlace place(0);
  for (PoolType failing_pool : {PoolType::kSmall, PoolType::kLarge}) {
    SCOPED_TRACE(static_cast<int>(failing_pool));
    auto bottom = std::make_shared<CUDAVirtualMemAllocatorV2>(
        place, 2UL << 20, failing_pool);
    // The place check throws before CUDA device selection, so this does not
    // require a second GPU or poison the CUDA context.
    auto invalid = std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
        bottom, 256, phi::GPUPlace(1), failing_pool);
    auto small = failing_pool == PoolType::kSmall
                     ? invalid
                     : CreatePoolAllocator(2UL << 20, PoolType::kSmall);
    auto large = failing_pool == PoolType::kLarge
                     ? invalid
                     : CreatePoolAllocator(2UL << 20, PoolType::kLarge);
    VMMAutoGrowthBestFitMultiPoolAllocatorV2 multi(
        small, large, 8UL << 20, place);
    const PoolType healthy_pool =
        failing_pool == PoolType::kSmall ? PoolType::kLarge : PoolType::kSmall;
    AllocationPtr anchor;
    PrepareRemapSource(&multi, &anchor, healthy_pool);
    auto healthy = healthy_pool == PoolType::kSmall ? small : large;
    void* source = healthy->all_blocks().front().ptr();
    ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
    EXPECT_THROW(multi.RemapForAllocation(place, 0),
                 common::enforce::EnforceNotMet);
    // Large runs first: its committed move survives a subsequent small-pool
    // error; a large-pool error prevents the small pool from running.
    EXPECT_TRUE(HasBlockWithPtr(*healthy,
                                source,
                                failing_pool == PoolType::kSmall
                                    ? BlockType::kUnmappedFree
                                    : BlockType::kFree));
    EXPECT_TRUE(HasBlockWithPtr(*healthy, anchor->ptr(), BlockType::kActive));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, ReleaseAggregatesPools) {
  auto allocator = CreateAllocator();

  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);

  small.reset();
  large.reset();

  EXPECT_EQ(allocator->Release(phi::GPUPlace()), 4UL << 20);
  EXPECT_TRUE(allocator->small_allocator()->all_blocks().empty());
  EXPECT_TRUE(allocator->large_allocator()->all_blocks().empty());
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2, CrossPoolAllocFree) {
  auto allocator = CreateAllocator();

  // Allocate across both routed pools.
  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);
  auto* small_ptr = small->ptr();
  auto* large_ptr = large->ptr();

  // Free in a different order than allocation.
  large.reset();
  auto large_reused = allocator->Allocate(2UL << 20);
  ASSERT_NE(large_reused, nullptr);
  EXPECT_EQ(large_reused->ptr(), large_ptr);

  small.reset();
  auto small_reused = allocator->Allocate(256UL);
  ASSERT_NE(small_reused, nullptr);
  EXPECT_EQ(small_reused->ptr(), small_ptr);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     CollectTensorPartsFindsRoutedV2Blocks) {
  auto allocator = CreateAllocator();

  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);

  std::vector<BlockPart> small_parts;
  ASSERT_TRUE(allocator->small_allocator()->CollectTensorParts(
      small->ptr(), small->size(), &small_parts));
  ASSERT_EQ(small_parts.size(), 1UL);
  EXPECT_EQ(small_parts[0].chunk_rel_off, 0UL);
  EXPECT_EQ(small_parts[0].len, small->size());

  std::vector<BlockPart> large_parts;
  ASSERT_TRUE(allocator->large_allocator()->CollectTensorParts(
      large->ptr(), large->size(), &large_parts));
  ASSERT_EQ(large_parts.size(), 1UL);
  EXPECT_EQ(large_parts[0].chunk_rel_off, 0UL);
  EXPECT_EQ(large_parts[0].len, large->size());
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     VmmTensorPartsVisitorFindsV2Blocks) {
  auto allocator = CreateAllocator();

  auto small = allocator->Allocate(256UL);
  auto large = allocator->Allocate(2UL << 20);
  ASSERT_NE(small, nullptr);
  ASSERT_NE(large, nullptr);

  paddle::memory::VmmTensorPartsVisitor small_visitor(
      small->ptr(), small->size(), false);
  allocator->Accept(&small_visitor);
  ASSERT_TRUE(small_visitor.Found());
  ASSERT_EQ(small_visitor.Parts().size(), 1UL);
  EXPECT_EQ(small_visitor.Parts()[0].len, small->size());
  EXPECT_EQ(small_visitor.Parts()[0].chunk->shared_fd, -1);

  paddle::memory::VmmTensorPartsVisitor large_visitor(
      large->ptr(), large->size(), false);
  allocator->Accept(&large_visitor);
  ASSERT_TRUE(large_visitor.Found());
  ASSERT_EQ(large_visitor.Parts().size(), 1UL);
  EXPECT_EQ(large_visitor.Parts()[0].len, large->size());
  EXPECT_EQ(large_visitor.Parts()[0].chunk->shared_fd, -1);

  allocator->Accept(&large_visitor);
  ASSERT_TRUE(large_visitor.Found());

  // A visitor that already found its allocation must remain a no-op when
  // another pool is visited directly.
  large_visitor.Visit(allocator->large_allocator().get());
  ASSERT_TRUE(large_visitor.Found());
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeRecordsRemapSafetyOnFree) {
  const phi::GPUPlace place(0);
  auto multi = std::shared_ptr<VMMAutoGrowthBestFitMultiPoolAllocatorV2>(
      CreateAllocator().release());
  auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
      multi, place, cudaStreamPerThread);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);

  auto allocation = stream_safe->Allocate(2UL << 20);
  auto anchor = stream_safe->Allocate(2UL << 20);
  ASSERT_NE(allocation, nullptr);
  ASSERT_NE(anchor, nullptr);
  allocation.reset();
  EXPECT_EQ(stream_safe->GetVMMV2Allocator(), multi.get());
  EXPECT_GT(stream_safe->Compact(place), 0UL);

  gpuStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  auto pending = stream_safe->Allocate(2UL << 20);
  auto* stream_safe_allocation =
      dynamic_cast<StreamSafeCUDAAllocation*>(pending.get());
  ASSERT_NE(stream_safe_allocation, nullptr);
  VMMStreamSafeBusyWait<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_TRUE(stream_safe_allocation->RecordStream(stream));
  pending.reset();
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_NO_THROW(stream_safe->Compact(place));
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  auto retry = std::make_shared<RetryAllocator>(multi, place, 0);
  auto retry_stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
      retry, place, cudaStreamPerThread);
  EXPECT_EQ(retry_stream_safe->GetVMMV2Allocator(), multi.get());

  auto stat = std::make_shared<StatAllocator>(multi);
  auto stat_stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
      stat, place, cudaStreamPerThread);
  EXPECT_EQ(stat_stream_safe->GetVMMV2Allocator(), multi.get());
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeOOMDispatchesByAllocatorType) {
  ScopedTemporaryCwd cwd;
  const phi::GPUPlace place(0);
  constexpr size_t kImpossibleSize = 1ULL << 50;
  ScopedVLogLevel vlog_guard(3);

  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>(
        /*requested_handles=*/2,
        /*created_handles=*/1,
        /*success_on_attempt=*/3);
    AllocationPtr anchor;
    PrepareRemapSource(multi.get(), &anchor);
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);

    auto allocation = stream_safe->Allocate(4UL << 20);
    ASSERT_NE(allocation, nullptr);
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
    EXPECT_EQ(stream_safe->release_calls(), 0UL);
  }
  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>();
    AllocationPtr anchor;
    PrepareRemapSource(multi.get(), &anchor);
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
    try {
      stream_safe->Allocate(4UL << 20);
      FAIL() << "Expected VMM V2 allocation to fail";
    } catch (const BadAlloc& ex) {
      const std::string message = ex.what();
      if (std::getenv("PADDLE_TEST_PRINT_VMM_V2_OOM") != nullptr) {
        std::cout << "OOM message sample:\n" << message << std::endl;
      }
      EXPECT_NE(message.find("Out of memory error on GPU"), std::string::npos);
      EXPECT_NE(message.find("Paddle allocator memory:"), std::string::npos);
      EXPECT_NE(message.find("Allocated (in use):"), std::string::npos);
      EXPECT_NE(message.find("Free in Paddle memory pool:"), std::string::npos);
      EXPECT_NE(message.find("Largest contiguous free block:"),
                std::string::npos);
      EXPECT_NE(message.find("CUDA driver memory:"), std::string::npos);
      EXPECT_NE(message.find("Free on device:"), std::string::npos);
      EXPECT_NE(message.find("Total device capacity:"), std::string::npos);
      EXPECT_NE(message.find("Memory defragmentation: remap moved"),
                std::string::npos);
      EXPECT_NE(message.find("Allocation Failure Summary:"), std::string::npos);
      EXPECT_EQ(message.find("Please stop other processes"), std::string::npos);
      EXPECT_NE(message.find("2. Retry after reclaiming pending frees:"),
                std::string::npos);
      EXPECT_NE(message.find("3. Retry after memory defragmentation:"),
                std::string::npos);
      EXPECT_NE(message.find("4. Retry after releasing memory from all "
                             "registered allocators:"),
                std::string::npos);
    }
    EXPECT_EQ(multi->allocation_attempts(), 4UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
  }
  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>();
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/false);
    try {
      stream_safe->Allocate(4UL << 20);
      FAIL() << "Expected VMM V2 allocation to fail";
    } catch (const BadAlloc& ex) {
      const std::string message = ex.what();
      EXPECT_NE(message.find("remap was not attempted because memory "
                             "defragmentation is disabled"),
                std::string::npos);
      EXPECT_NE(message.find("2. Retry after reclaiming pending frees:"),
                std::string::npos);
    }
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
  }
  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>();
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
    try {
      stream_safe->Allocate(1UL << 20);
      FAIL() << "Expected the small-pool allocation to fail";
    } catch (const BadAlloc& ex) {
      const std::string message = ex.what();
      EXPECT_NE(message.find("remap was not attempted because no useful "
                             "remap work was identified"),
                std::string::npos);
      EXPECT_NE(message.find("3. Retry after releasing memory from all "
                             "registered allocators:"),
                std::string::npos);
      EXPECT_EQ(message.find("4. Retry"), std::string::npos);
    }
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
  }
  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>(
        /*requested_handles=*/2, /*created_handles=*/0);
    AllocationPtr anchor;
    PrepareRemapSource(multi.get(), &anchor);
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
    try {
      stream_safe->Allocate(4UL << 20);
      FAIL() << "Expected VMM V2 allocation to fail";
    } catch (const BadAlloc& ex) {
      const std::string message = ex.what();
      EXPECT_NE(message.find("remap was not attempted; 2.000000MB could be "
                             "safely moved, but 4.000000MB was required"),
                std::string::npos);
    }
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
  }
  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>();
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
    try {
      stream_safe->Allocate(4UL << 20);
      FAIL() << "Expected VMM V2 allocation to fail";
    } catch (const BadAlloc& ex) {
      const std::string message = ex.what();
      EXPECT_NE(message.find("remap was not attempted because no safely "
                             "movable free memory was available"),
                std::string::npos);
    }
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
  }
  {
    auto cuda_allocator = std::make_shared<CUDAAllocator>(place);
    auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
        cuda_allocator, place, cudaStreamPerThread);
    EXPECT_THROW(stream_safe->Allocate(kImpossibleSize), BadAlloc);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeSmallPoolRemapsWithLargerThreshold) {
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  // A request spanning multiple handles must still route to the small pool
  // to exercise an actual remap, rather than the reusable-block precheck.
  for (size_t success_on_attempt : {3UL, 4UL}) {
    SCOPED_TRACE(success_on_attempt);
    auto multi = std::make_shared<FailingMultiPoolAllocator>(
        /*requested_handles=*/2,
        /*created_handles=*/1,
        success_on_attempt,
        /*small_allocation_threshold=*/8UL << 20);
    AllocationPtr anchor;
    PrepareRemapSource(multi.get(), &anchor, PoolType::kSmall);
    void* source_ptr = multi->small_allocator()->all_blocks().front().ptr();
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);

    auto allocation = stream_safe->Allocate(4UL << 20);
    ASSERT_NE(dynamic_cast<StreamSafeCUDAAllocation*>(allocation.get()),
              nullptr);
    EXPECT_EQ(multi->allocation_attempts(), success_on_attempt);
    EXPECT_EQ(stream_safe->release_calls(),
              success_on_attempt == 4 ? 1UL : 0UL);
    EXPECT_TRUE(HasBlockWithPtr(
        *multi->small_allocator(), source_ptr, BlockType::kUnmappedFree));
    EXPECT_TRUE(HasBlockWithPtr(
        *multi->small_allocator(), anchor->ptr(), BlockType::kActive));
    EXPECT_TRUE(HasBlockWithPtr(
        *multi->small_allocator(), allocation->ptr(), BlockType::kActive));
    EXPECT_TRUE(multi->large_allocator()->all_blocks().empty());
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeSmallPoolOOMReleasesLargePool) {
  ScopedTemporaryCwd cwd;
  const phi::GPUPlace place(0);
  const std::pair<size_t, bool> configurations[] = {{1UL << 20, false},
                                                    {1UL << 20, true},
                                                    {2UL << 20, false},
                                                    {2UL << 20, true}};
  for (const auto& [threshold, remap_on_oom] : configurations) {
    SCOPED_TRACE(threshold);
    SCOPED_TRACE(remap_on_oom);
    const size_t block_size = threshold / 2;
    const size_t request_size = threshold * 3 / 4;
    ScopedVMMRetryFlags flags(/*retry_times=*/0, remap_on_oom);
    for (size_t success_on_attempt : {0UL, 3UL}) {
      SCOPED_TRACE(success_on_attempt);
      auto multi = std::make_shared<FailingMultiPoolAllocator>(
          /*requested_handles=*/1,
          /*created_handles=*/0,
          success_on_attempt,
          threshold);
      auto stream_safe =
          std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
              multi, place, cudaStreamPerThread);

      // Live anchors keep the free fragments smaller than the request and
      // prevent releasing or remapping any whole small-pool handle.
      auto first = multi->small_allocator()->Allocate(block_size);
      auto first_anchor = multi->small_allocator()->Allocate(block_size);
      auto second = multi->small_allocator()->Allocate(block_size);
      auto second_anchor = multi->small_allocator()->Allocate(block_size);
      first.reset();
      second.reset();
      size_t total_free = 0;
      size_t largest_free = 0;
      multi->GetFreeBlockStats(&total_free, &largest_free, request_size);
      ASSERT_EQ(total_free, threshold);
      ASSERT_EQ(largest_free, block_size);

      auto large_cache = multi->large_allocator()->Allocate(2UL << 20);
      auto large_anchor = multi->large_allocator()->Allocate(2UL << 20);
      void* large_cache_ptr = large_cache->ptr();
      large_cache.reset();
      ASSERT_TRUE(HasBlockWithPtr(
          *multi->large_allocator(), large_cache_ptr, BlockType::kFree));

      auto peer_multi =
          std::shared_ptr<VMMAutoGrowthBestFitMultiPoolAllocatorV2>(
              CreateAllocator().release());
      auto peer_stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
          peer_multi, place, cudaStreamPerThread);
      auto peer_large = peer_stream_safe->Allocate(2UL << 20);
      peer_large.reset();
      ASSERT_FALSE(peer_multi->large_allocator()->all_blocks().empty());

      AllocationPtr allocation;
      if (success_on_attempt != 0) {
        allocation = stream_safe->Allocate(request_size);
        ASSERT_NE(dynamic_cast<StreamSafeCUDAAllocation*>(allocation.get()),
                  nullptr);
        EXPECT_TRUE(HasBlockWithPtr(
            *multi->small_allocator(), allocation->ptr(), BlockType::kActive));
      } else {
        try {
          stream_safe->Allocate(request_size);
          FAIL() << "Expected the final small-pool allocation attempt to fail";
        } catch (const BadAlloc& ex) {
          const std::string message = ex.what();
          EXPECT_NE(
              message.find(remap_on_oom ? "remap was not attempted because no "
                                          "useful remap work was identified"
                                        : "defragmentation is disabled"),
              std::string::npos);
          EXPECT_EQ(message.find("final allocation retry still failed"),
                    std::string::npos);
          EXPECT_NE(message.find("3. Retry after releasing memory from all "
                                 "registered allocators:"),
                    std::string::npos);
          EXPECT_EQ(message.find("4. Retry"), std::string::npos);
        }
      }
      EXPECT_EQ(multi->allocation_attempts(), 3UL);
      EXPECT_EQ(stream_safe->release_calls(), 1UL);
      EXPECT_EQ(stream_safe->released_bytes(), 4UL << 20);
      EXPECT_TRUE(HasBlockWithPtr(*multi->large_allocator(),
                                  large_cache_ptr,
                                  BlockType::kUnmappedFree));
      EXPECT_TRUE(peer_multi->large_allocator()->all_blocks().empty());
      EXPECT_TRUE(HasBlockWithPtr(
          *multi->large_allocator(), large_anchor->ptr(), BlockType::kActive));
      EXPECT_TRUE(HasBlockWithPtr(
          *multi->small_allocator(), first_anchor->ptr(), BlockType::kActive));
      EXPECT_TRUE(HasBlockWithPtr(
          *multi->small_allocator(), second_anchor->ptr(), BlockType::kActive));
      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeSmallPoolRetriesAfterZeroByteRelease) {
  ScopedTemporaryCwd cwd;
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  for (size_t threshold : {1UL << 20, 2UL << 20}) {
    SCOPED_TRACE(threshold);
    for (size_t success_on_attempt : {0UL, 3UL}) {
      SCOPED_TRACE(success_on_attempt);
      auto multi = std::make_shared<FailingMultiPoolAllocator>(
          /*requested_handles=*/1,
          /*created_handles=*/0,
          success_on_attempt,
          threshold);
      auto stream_safe =
          std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
              multi, place, cudaStreamPerThread);
      // Both pools are empty, so the real coordinated release returns zero.
      if (success_on_attempt != 0) {
        auto allocation = stream_safe->Allocate(threshold / 2);
        EXPECT_NE(dynamic_cast<StreamSafeCUDAAllocation*>(allocation.get()),
                  nullptr);
      } else {
        EXPECT_THROW(stream_safe->Allocate(threshold / 2), BadAlloc);
      }
      EXPECT_EQ(multi->allocation_attempts(), 3UL);
      EXPECT_EQ(stream_safe->release_calls(), 1UL);
      EXPECT_EQ(stream_safe->released_bytes(), 0UL);
      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeRemapExceptionStopsRecovery) {
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  for (PoolType failing_pool : {PoolType::kSmall, PoolType::kLarge}) {
    SCOPED_TRACE(static_cast<int>(failing_pool));
    auto bottom = std::make_shared<CUDAVirtualMemAllocatorV2>(
        place, 2UL << 20, failing_pool);
    auto invalid = std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
        bottom, 256, phi::GPUPlace(1), failing_pool);
    auto multi = std::make_shared<FailingMultiPoolAllocator>(
        failing_pool == PoolType::kSmall
            ? invalid
            : CreatePoolAllocator(2UL << 20, PoolType::kSmall),
        failing_pool == PoolType::kLarge
            ? invalid
            : CreatePoolAllocator(2UL << 20, PoolType::kLarge),
        /*requested_handles=*/2,
        /*created_handles=*/1,
        /*success_on_attempt=*/0,
        /*small_allocation_threshold=*/8UL << 20);
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    const size_t request =
        failing_pool == PoolType::kSmall ? 4UL << 20 : 8UL << 20;
    try {
      stream_safe->Allocate(request);
      FAIL() << "Expected the remap place check to fail";
    } catch (const common::enforce::EnforceNotMet& ex) {
      EXPECT_NE(std::string(ex.what()).find("does not match allocator place"),
                std::string::npos);
    }
    EXPECT_EQ(multi->allocation_attempts(), 2UL);
    EXPECT_EQ(stream_safe->release_calls(), 0UL);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeReleaseExceptionStopsFinalRetry) {
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  auto bottom = std::make_shared<FailingUnmapAllocator>();
  auto large = std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
      bottom, 256, place, PoolType::kLarge);
  auto multi = std::make_shared<FailingMultiPoolAllocator>(
      CreatePoolAllocator(2UL << 20, PoolType::kSmall), large);
  auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
      multi, place, cudaStreamPerThread);
  auto cache = large->Allocate(2UL << 20);
  void* cache_ptr = cache->ptr();
  cache.reset();
  bottom->FailNextUnmap();
  EXPECT_THROW(stream_safe->Allocate(1UL << 20),
               common::enforce::EnforceNotMet);
  EXPECT_EQ(multi->allocation_attempts(), 2UL);
  EXPECT_EQ(stream_safe->release_calls(), 1UL);
  EXPECT_EQ(bottom->unmap_calls(), 1UL);
  EXPECT_TRUE(HasBlockWithPtr(*large, cache_ptr, BlockType::kFree));
  EXPECT_FALSE(bottom->IsRangeUnmapped(
      reinterpret_cast<VMMDevicePtr>(cache_ptr), 2UL << 20));
  // The failed unmap preserves backing and unlocks the registry and pool.
  EXPECT_EQ(stream_safe->Release(place), 2UL << 20);
  EXPECT_TRUE(large->all_blocks().empty());
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeSynchronizationExceptionStopsFinalRetry) {
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  auto bottom = std::make_shared<FailingUnmapAllocator>();
  auto large = std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
      bottom, 256, place, PoolType::kLarge);
  auto multi = std::make_shared<FailingMultiPoolAllocator>(
      CreatePoolAllocator(2UL << 20, PoolType::kSmall), large);
  auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
      multi, place, cudaStreamPerThread);
  auto cache = large->Allocate(2UL << 20);
  void* cache_ptr = cache->ptr();
  cache.reset();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  {
    gpuStream_t capture_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking),
              cudaSuccess);
    bool capturing = false;
    DEFINE_PADDLE_SCOPE_GUARD([&] {
      if (capturing) {
        cudaGraph_t graph = nullptr;
        const auto status = cudaStreamEndCapture(capture_stream, &graph);
        EXPECT_TRUE(status == cudaSuccess ||
                    status == cudaErrorStreamCaptureInvalidated);
        if (graph != nullptr) {
          EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        }
      }
      EXPECT_EQ(cudaStreamDestroy(capture_stream), cudaSuccess);
      cudaGetLastError();
    });
    // Raw global capture makes the real cudaDeviceSynchronize fail without
    // an illegal-access kernel or a permanently damaged CUDA context.
    ASSERT_EQ(
        cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal),
        cudaSuccess);
    capturing = true;
    EXPECT_THROW(stream_safe->Allocate(1UL << 20),
                 common::enforce::EnforceNotMet);
    EXPECT_EQ(multi->allocation_attempts(), 2UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
    EXPECT_EQ(bottom->unmap_calls(), 0UL);
    EXPECT_TRUE(HasBlockWithPtr(*large, cache_ptr, BlockType::kFree));
  }
  EXPECT_EQ(stream_safe->Release(place), 2UL << 20);
  EXPECT_TRUE(large->all_blocks().empty());
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeReleaseAllRecoversAfterRemapRetryFailure) {
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);

  for (bool report_zero_released_bytes : {false, true}) {
    SCOPED_TRACE(report_zero_released_bytes);
    auto multi = std::make_shared<FailingMultiPoolAllocator>(
        /*requested_handles=*/2,
        /*created_handles=*/1,
        /*success_on_attempt=*/4);
    AllocationPtr anchor;
    PrepareRemapSource(multi.get(), &anchor);
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    if (report_zero_released_bytes) {
      stream_safe->ReportZeroReleasedBytes();
    }

    auto peer_multi = std::shared_ptr<VMMAutoGrowthBestFitMultiPoolAllocatorV2>(
        CreateAllocator().release());
    auto peer_stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
        peer_multi, place, cudaStreamPerThread);
    auto peer_small = peer_stream_safe->Allocate(256UL);
    auto peer_large = peer_stream_safe->Allocate(2UL << 20);
    peer_small.reset();
    peer_large.reset();
    ASSERT_FALSE(peer_multi->small_allocator()->all_blocks().empty());
    ASSERT_FALSE(peer_multi->large_allocator()->all_blocks().empty());

    bool release_completed = false;
    stream_safe->SetReleaseObserver([&] {
      EXPECT_EQ(multi->allocation_attempts(), 3UL);
      EXPECT_TRUE(peer_multi->small_allocator()->all_blocks().empty());
      EXPECT_TRUE(peer_multi->large_allocator()->all_blocks().empty());
      release_completed = true;
    });
    multi->SetAllocationObserver(
        [&](size_t attempt) { EXPECT_EQ(release_completed, attempt == 4); });
    auto allocation = stream_safe->Allocate(4UL << 20);
    ASSERT_NE(allocation, nullptr);
    auto* stream_safe_allocation =
        dynamic_cast<StreamSafeCUDAAllocation*>(allocation.get());
    ASSERT_NE(stream_safe_allocation, nullptr);
    EXPECT_EQ(stream_safe_allocation->GetOwningStream(), cudaStreamPerThread);
    EXPECT_EQ(allocation->size(), 4UL << 20);
    EXPECT_EQ(multi->allocation_attempts(), 4UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
    EXPECT_GT(stream_safe->released_bytes(), 0UL);
    EXPECT_TRUE(peer_multi->small_allocator()->all_blocks().empty());
    EXPECT_TRUE(peer_multi->large_allocator()->all_blocks().empty());
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeReleaseAfterRemapMakesNoProgress) {
  ScopedTemporaryCwd cwd;
  const phi::GPUPlace place(0);
  constexpr size_t kHandleSize = 2UL << 20;
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  for (PoolType pool_type : {PoolType::kSmall, PoolType::kLarge}) {
    SCOPED_TRACE(static_cast<int>(pool_type));
    for (size_t success_on_attempt : {0UL, 3UL}) {
      SCOPED_TRACE(success_on_attempt);
      auto bottom = std::make_shared<CUDAVirtualMemAllocatorV2>(
          place, kHandleSize, pool_type);
      auto blocked = std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
          bottom, 256, place, pool_type);
      auto multi = std::make_shared<FailingMultiPoolAllocator>(
          pool_type == PoolType::kSmall
              ? blocked
              : CreatePoolAllocator(kHandleSize, PoolType::kSmall),
          pool_type == PoolType::kLarge
              ? blocked
              : CreatePoolAllocator(kHandleSize, PoolType::kLarge),
          2,
          1,
          success_on_attempt,
          pool_type == PoolType::kSmall ? 8UL << 20 : kHandleSize);
      AllocationPtr anchor;
      PrepareRemapSource(multi.get(), &anchor, pool_type);
      auto hidden_tail = bottom->AppendWithBlock(kHandleSize);
      ASSERT_TRUE(hidden_tail.HasAllocation());
      ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
      const auto before = blocked->SnapshotBlockInfo();
      auto stream_safe =
          std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
              multi, place, cudaStreamPerThread);
      bool released = false;
      multi->SetAllocationObserver(
          [&](size_t attempt) { EXPECT_EQ(released, attempt == 3); });
      stream_safe->SetBeforeReleaseObserver([&] {
        EXPECT_EQ(multi->allocation_attempts(), 2UL);
        EXPECT_EQ(blocked->SnapshotBlockInfo(), before);
      });
      stream_safe->SetReleaseObserver([&] {
        EXPECT_EQ(multi->allocation_attempts(), 2UL);
        // Remove the synthetic placement barrier before the final real grow.
        hidden_tail.allocation.reset();
        released = true;
      });
      if (success_on_attempt != 0) {
        auto allocation = stream_safe->Allocate(4UL << 20);
        auto* wrapped =
            dynamic_cast<StreamSafeCUDAAllocation*>(allocation.get());
        ASSERT_NE(wrapped, nullptr);
        EXPECT_EQ(wrapped->GetOwningStream(), cudaStreamPerThread);
        EXPECT_EQ(wrapped->size(), 4UL << 20);
      } else {
        try {
          stream_safe->Allocate(4UL << 20);
          FAIL()
              << "Expected final allocation failure after zero-progress remap";
        } catch (const BadAlloc& ex) {
          const std::string message = ex.what();
          EXPECT_NE(
              message.find("remap was attempted, but no memory was moved"),
              std::string::npos);
          EXPECT_NE(message.find("3. Retry after releasing memory from all "
                                 "registered allocators:"),
                    std::string::npos);
          EXPECT_EQ(message.find("3. Retry after memory defragmentation:"),
                    std::string::npos);
          EXPECT_EQ(message.find("4. Retry"), std::string::npos);
        }
      }
      EXPECT_EQ(multi->allocation_attempts(), 3UL);
      EXPECT_EQ(stream_safe->release_calls(), 1UL);
      EXPECT_EQ(stream_safe->released_bytes(), kHandleSize);
      EXPECT_TRUE(HasBlockWithPtr(*blocked, anchor->ptr(), BlockType::kActive));
      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeReleaseRecoversWhenRemapIsUnavailable) {
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  for (bool insufficient_movable : {false, true}) {
    SCOPED_TRACE(insufficient_movable);
    auto multi = std::make_shared<FailingMultiPoolAllocator>(
        /*requested_handles=*/2,
        /*created_handles=*/0,
        /*success_on_attempt=*/3);
    AllocationPtr anchor;
    if (insufficient_movable) {
      PrepareRemapSource(multi.get(), &anchor);
    }
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    bool released = false;
    stream_safe->SetReleaseObserver([&] {
      EXPECT_EQ(multi->allocation_attempts(), 2UL);
      released = true;
    });
    multi->SetAllocationObserver(
        [&](size_t attempt) { EXPECT_EQ(released, attempt == 3); });
    auto allocation = stream_safe->Allocate(4UL << 20);
    ASSERT_NE(dynamic_cast<StreamSafeCUDAAllocation*>(allocation.get()),
              nullptr);
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
    EXPECT_EQ(stream_safe->released_bytes(),
              insufficient_movable ? 2UL << 20 : 0UL);
    EXPECT_TRUE(HasBlockWithPtr(
        *multi->large_allocator(), allocation->ptr(), BlockType::kActive));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeFinalOOMUsesStatisticsAfterLastAttempt) {
  ScopedTemporaryCwd cwd;
  const phi::GPUPlace place(0);
  for (bool remap_on_oom : {false, true}) {
    SCOPED_TRACE(remap_on_oom);
    ScopedVMMRetryFlags flags(/*retry_times=*/0, remap_on_oom);
    auto multi = std::make_shared<FailingMultiPoolAllocator>();
    AllocationPtr anchor;
    PrepareRemapSource(multi.get(), &anchor);
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    const size_t final_attempt = remap_on_oom ? 4 : 3;
    std::array<AllocationPtr, 2> small_anchors;
    AllocationPtr large_boundary_anchor;
    bool released = false;
    stream_safe->SetReleaseObserver([&] {
      EXPECT_EQ(multi->allocation_attempts(), final_attempt - 1);
      size_t total_free = 1;
      size_t largest_free = 1;
      multi->GetFreeBlockStats(&total_free, &largest_free, 4UL << 20);
      EXPECT_EQ(total_free, 0UL);
      EXPECT_EQ(largest_free, 0UL);
      released = true;
    });
    multi->SetAllocationObserver([&](size_t attempt) {
      EXPECT_EQ(released, attempt == final_attempt);
      if (attempt == final_attempt) {
        // The failing final attempt populates both pools and leaves new
        // backing cached. The dump and tables must reflect this final state.
        // Exact bucket boundaries belong to [1M,10M) and [10M,50M).
        for (auto& small_anchor : small_anchors) {
          small_anchor = multi->small_allocator()->Allocate(1UL << 20);
        }
        large_boundary_anchor = multi->large_allocator()->Allocate(10UL << 20);
        auto cached = multi->large_allocator()->Allocate(6UL << 20);
        cached.reset();
      }
    });
    std::string message;
    try {
      stream_safe->Allocate(4UL << 20);
      FAIL() << "Expected the injected final allocation failure";
    } catch (const BadAlloc& ex) {
      message = ex.what();
      EXPECT_NE(message.find("Free in Paddle memory pool: 6.000000MB"),
                std::string::npos);
      EXPECT_NE(message.find("Largest contiguous free block: 6.000000MB"),
                std::string::npos);
      EXPECT_NE(message.find(std::to_string(final_attempt) +
                             ". Retry after releasing memory from all "
                             "registered allocators:"),
                std::string::npos);
      EXPECT_NE(
          message.find("deterministic VMM stream-safe allocation failure " +
                       std::to_string(final_attempt)),
          std::string::npos);
      EXPECT_NE(
          message.find("all_block_info summary: groups=2 (group 0=small pool, "
                       "group 1=large pool), blocks=5, mapped=20.00 MB, "
                       "used=14.00 MB, free=6.00 MB, free_blocks=1, "
                       "max_free=6.00 MB\n"),
          std::string::npos)
          << message;
      ExpectMemorySummary(message);
      const auto compact = RemoveWhitespace(message);
      for (const char* row :
           {"|Allocator_0|2|2|0|2.00MB|2.00MB|0B|1.00MB|0B|",
            "|Allocator_1|3|2|1|18.00MB|12.00MB|6.00MB|10.00MB|6.00MB|",
            "|Allocator_0|FreeBlocks|0|0|0|0|0|0|0|0|0|0|0|0|",
            "|Allocator_0|UsedBlocks|0|2|0|0|0|0|0|0|0|0|0|0|",
            "|Allocator_1|FreeBlocks|0|1|0|0|0|0|0|0|0|0|0|0|",
            "|Allocator_1|UsedBlocks|0|1|1|0|0|0|0|0|0|0|0|0|"}) {
        EXPECT_NE(compact.find(row), std::string::npos) << row;
      }
    }
    const auto small_blocks = multi->small_allocator()->SnapshotBlockInfo();
    const auto large_blocks = multi->large_allocator()->SnapshotBlockInfo();
    ASSERT_EQ(small_blocks.size(), 2UL);
    ASSERT_EQ(large_blocks.size(), 3UL);
    ExpectBlockInfoDump(message, {small_blocks, large_blocks});
    EXPECT_EQ(multi->allocation_attempts(), final_attempt);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
  // Each final OOM writes its own file instead of replacing the previous one.
  EXPECT_EQ(DumpFiles().size(), 2UL);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeRecoveryDoesNotDumpBlockInfo) {
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  for (size_t success_on_attempt : {2UL, 3UL, 4UL}) {
    SCOPED_TRACE(success_on_attempt);
    ScopedTemporaryCwd cwd;
    auto multi = std::make_shared<FailingMultiPoolAllocator>(
        /*requested_handles=*/2, /*created_handles=*/1, success_on_attempt);
    AllocationPtr anchor;
    PrepareRemapSource(multi.get(), &anchor);
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    multi->SetAllocationObserver(
        [](size_t) { EXPECT_TRUE(DumpFiles().empty()); });
    auto allocation = stream_safe->Allocate(4UL << 20);
    ASSERT_NE(dynamic_cast<StreamSafeCUDAAllocation*>(allocation.get()),
              nullptr);
    EXPECT_EQ(multi->allocation_attempts(), success_on_attempt);
    EXPECT_EQ(stream_safe->release_calls(),
              success_on_attempt == 4 ? 1UL : 0UL);
    EXPECT_TRUE(std::filesystem::is_empty("."));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeFinalOOMEmptySnapshotKeepsEarlierDumps) {
  ScopedTemporaryCwd cwd;
  const phi::GPUPlace place(0);
  // A file using the old fixed name must not be touched.
  {
    std::ofstream legacy("mem_info.core");
    legacy << "legacy dump";
    ASSERT_TRUE(legacy.good());
  }
  std::vector<std::filesystem::path> saved;
  for (bool remap_on_oom : {false, true}) {
    SCOPED_TRACE(remap_on_oom);
    ScopedVMMRetryFlags flags(/*retry_times=*/0, remap_on_oom);
    auto multi = std::make_shared<FailingMultiPoolAllocator>();
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);
    std::string message;
    try {
      stream_safe->Allocate(4UL << 20);
      FAIL() << "Expected final OOM with empty pools";
    } catch (const BadAlloc& ex) {
      message = ex.what();
      ExpectMemorySummary(message);
      EXPECT_NE(message.find("all_block_info summary: groups=0 (no mapped "
                             "pool), blocks=0, mapped=0 B, used=0 B, free=0 "
                             "B, free_blocks=0, max_free=0 B\n"),
                std::string::npos)
          << message;
      EXPECT_EQ(message.find("Failed to write all_block_info"),
                std::string::npos);
      EXPECT_EQ(message.find("Allocator_0"), std::string::npos);
    }
    EXPECT_TRUE(multi->small_allocator()->SnapshotBlockInfo().empty());
    EXPECT_TRUE(multi->large_allocator()->SnapshotBlockInfo().empty());
    ExpectBlockInfoDump(message, {});
    saved.push_back(SavedDumpPath(message).filename());
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
  ASSERT_EQ(saved.size(), 2UL);
  EXPECT_NE(saved[0], saved[1]);
  auto expected = saved;
  expected.push_back("mem_info.core");
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(DumpFiles(), expected);
  EXPECT_EQ(ReadTextFile("mem_info.core"), "legacy dump");
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeFinalOOMDumpWriteFailurePreservesBadAlloc) {
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/false);
  auto multi = std::make_shared<FailingMultiPoolAllocator>();
  auto anchor = multi->small_allocator()->Allocate(2UL << 20);
  auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
      multi, place, cudaStreamPerThread);
  ScopedTemporaryCwd cwd;
  const auto workdir = std::filesystem::current_path();
  // Make the dump unwritable: the working directory no longer exists.
  ASSERT_TRUE(std::filesystem::remove(workdir));
  std::string message;
  try {
    stream_safe->Allocate(4UL << 20);
    FAIL() << "Expected BadAlloc even when the dump cannot be written";
  } catch (const BadAlloc& ex) {
    message = ex.what();
  }
  ASSERT_TRUE(std::filesystem::create_directory(workdir));
  ExpectMemorySummary(message);
  EXPECT_NE(message.find("Failed to write all_block_info to "),
            std::string::npos);
  EXPECT_NE(message.find("mem_info.core."), std::string::npos);
  EXPECT_EQ(message.find("all_block_info saved to"), std::string::npos);
  EXPECT_NE(message.find("all_block_info summary: groups=1 (group 0=small "
                         "pool), blocks=1, mapped=2.00 MB, used=2.00 MB"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("deterministic VMM stream-safe allocation failure 3"),
            std::string::npos);
  EXPECT_NE(message.find("3. Retry after releasing memory from all "
                         "registered allocators:"),
            std::string::npos);
  EXPECT_NE(RemoveWhitespace(message).find(
                "|Allocator_0|1|1|0|2.00MB|2.00MB|0B|2.00MB|0B|"),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::is_empty(workdir));
  EXPECT_EQ(multi->allocation_attempts(), 3UL);
  EXPECT_EQ(stream_safe->release_calls(), 1UL);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeReleaseAllFailurePreservesOOMSummary) {
  ScopedTemporaryCwd cwd;
  const phi::GPUPlace place(0);
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  auto multi = std::make_shared<FailingMultiPoolAllocator>();
  AllocationPtr anchor;
  PrepareRemapSource(multi.get(), &anchor);
  auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
      multi, place, cudaStreamPerThread);

  try {
    stream_safe->Allocate(4UL << 20);
    FAIL() << "Expected the final allocation attempt to fail";
  } catch (const BadAlloc& ex) {
    const std::string message = ex.what();
    EXPECT_NE(message.find("Paddle allocator memory:"), std::string::npos);
    EXPECT_NE(message.find("Allocated (in use):"), std::string::npos);
    EXPECT_NE(message.find("Reserved by Paddle:"), std::string::npos);
    EXPECT_NE(message.find("Free in Paddle memory pool:"), std::string::npos);
    EXPECT_NE(message.find("Largest contiguous free block:"),
              std::string::npos);
    EXPECT_NE(message.find("CUDA driver memory:"), std::string::npos);
    EXPECT_NE(message.find("Memory defragmentation: remap moved"),
              std::string::npos);
    // The numbered failure summary already reports the release retry.
    EXPECT_EQ(message.find("Memory release from all registered allocators"),
              std::string::npos);

    const char* stages[] = {
        "1. Initial attempt:",
        "2. Retry after reclaiming pending frees:",
        "3. Retry after memory defragmentation:",
        "4. Retry after releasing memory from all registered allocators:"};
    size_t previous_end = 0;
    for (size_t i = 0; i < 4; ++i) {
      const auto stage = message.find(stages[i]);
      ASSERT_NE(stage, std::string::npos);
      EXPECT_GE(stage, previous_end);
      const auto summary = "deterministic VMM stream-safe allocation failure " +
                           std::to_string(i + 1);
      const auto failure = message.find(summary, stage);
      ASSERT_NE(failure, std::string::npos);
      previous_end = failure + summary.size();
    }
  }
  EXPECT_EQ(multi->allocation_attempts(), 4UL);
  EXPECT_EQ(stream_safe->release_calls(), 1UL);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeRetriesWhenFreeBlockAppearsBeforeRemap) {
  ScopedTemporaryCwd cwd;
  const phi::GPUPlace place(0);
  constexpr size_t kRequestSize = 4UL << 20;
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);
  ScopedVLogLevel vlog_guard(3);

  for (size_t success_on_attempt : {3UL, 4UL}) {
    SCOPED_TRACE(success_on_attempt);
    auto multi = std::make_shared<RetryWithoutRemapMultiPoolAllocator>(
        success_on_attempt);
    multi->PrepareRetryBlock(kRequestSize);
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);

    stream_safe->SetReleaseObserver([&] {
      EXPECT_EQ(multi->allocation_attempts(), 3UL);
      EXPECT_TRUE(multi->large_allocator()->all_blocks().empty());
    });
    auto allocation = stream_safe->Allocate(kRequestSize);
    ASSERT_NE(allocation, nullptr);
    auto* wrapped = dynamic_cast<StreamSafeCUDAAllocation*>(allocation.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->GetOwningStream(), cudaStreamPerThread);
    EXPECT_EQ(wrapped->size(), kRequestSize);
    EXPECT_TRUE(HasBlockWithPtr(
        *multi->large_allocator(), wrapped->ptr(), BlockType::kActive));
    EXPECT_EQ(multi->allocation_attempts(), success_on_attempt);
    EXPECT_EQ(stream_safe->release_calls(),
              success_on_attempt == 4 ? 1UL : 0UL);
    EXPECT_EQ(stream_safe->released_bytes(),
              success_on_attempt == 4 ? kRequestSize : 0UL);
  }

  {
    auto multi = std::make_shared<RetryWithoutRemapMultiPoolAllocator>(0);
    multi->PrepareRetryBlock(kRequestSize);
    auto stream_safe = std::make_shared<ReleaseCountingStreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);

    try {
      stream_safe->Allocate(kRequestSize);
      FAIL() << "Expected the final allocation attempt to fail";
    } catch (const BadAlloc& ex) {
      const std::string message = ex.what();
      EXPECT_NE(message.find("remap was skipped because a sufficiently large "
                             "free block was available"),
                std::string::npos);
      EXPECT_NE(
          message.find("3. Retry after detecting an available free block:"),
          std::string::npos);
      EXPECT_NE(message.find("4. Retry after releasing memory from all "
                             "registered allocators:"),
                std::string::npos);
      EXPECT_EQ(message.find("Memory release from all registered allocators"),
                std::string::npos);
      EXPECT_EQ(message.find("remap moved 0.000000B"), std::string::npos);
      EXPECT_NE(message.find("Free in Paddle memory pool: 0.000000B"),
                std::string::npos);
      EXPECT_NE(message.find("Largest contiguous free block: 0.000000B"),
                std::string::npos);
      const char* stages[] = {
          "1. Initial attempt:",
          "2. Retry after reclaiming pending frees:",
          "3. Retry after detecting an available free block:",
          "4. Retry after releasing memory from all registered allocators:"};
      size_t previous_end = 0;
      for (size_t i = 0; i < 4; ++i) {
        const auto stage = message.find(stages[i]);
        ASSERT_NE(stage, std::string::npos);
        EXPECT_GE(stage, previous_end);
        const auto summary =
            "deterministic concurrent-free allocation failure " +
            std::to_string(i + 1);
        const auto failure = message.find(summary, stage);
        ASSERT_NE(failure, std::string::npos);
        previous_end = failure + summary.size();
      }
    }
    EXPECT_EQ(multi->allocation_attempts(), 4UL);
    EXPECT_EQ(stream_safe->release_calls(), 1UL);
    EXPECT_EQ(stream_safe->released_bytes(), kRequestSize);
    EXPECT_TRUE(multi->large_allocator()->all_blocks().empty());
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     ConcurrentAllocationMaintenanceAndReleasePreserveData) {
  const phi::GPUPlace place(0);
  constexpr size_t kIterations = 32;
  for (size_t threshold : {1UL << 20, 2UL << 20, 8UL << 20}) {
    SCOPED_TRACE(threshold);
    auto multi = std::make_shared<VMMAutoGrowthBestFitMultiPoolAllocatorV2>(
        CreatePoolAllocator(2UL << 20, PoolType::kSmall),
        CreatePoolAllocator(2UL << 20, PoolType::kLarge),
        threshold,
        place);
    std::array<gpuStream_t, 2> streams{};
    DEFINE_PADDLE_SCOPE_GUARD([&] {
      for (auto stream : streams) {
        if (stream != nullptr) {
          EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
      }
    });
    std::array<std::shared_ptr<StreamSafeCUDAAllocator>, 2> allocators;
    for (size_t i = 0; i < streams.size(); ++i) {
      ASSERT_EQ(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking),
                cudaSuccess);
      allocators[i] =
          std::make_shared<StreamSafeCUDAAllocator>(multi, place, streams[i]);
    }
    std::atomic<bool> start{false};
    std::array<std::exception_ptr, 3> errors{};
    std::array<size_t, 3> completed{};
    std::array<std::thread, 3> threads;
    for (size_t worker = 0; worker < threads.size(); ++worker) {
      threads[worker] = std::thread([&, worker] {
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        try {
          PADDLE_ENFORCE_GPU_SUCCESS(cudaSetDevice(0));
          for (size_t iteration = 0; iteration < kIterations; ++iteration) {
            if (worker == 2) {
              allocators[0]->Compact(place);
              allocators[0]->Release(place);
            } else {
              auto small = allocators[worker]->Allocate(256UL << 10);
              auto large = allocators[worker]->Allocate(8UL << 20);
              const auto value = static_cast<unsigned char>(
                  worker * kIterations + iteration + 1);
              for (const auto* allocation : {small.get(), large.get()}) {
                PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(allocation->ptr(),
                                                           value,
                                                           allocation->size(),
                                                           streams[worker]));
              }
              for (const auto* allocation : {small.get(), large.get()}) {
                std::array<unsigned char, 64> data{};
                PADDLE_ENFORCE_GPU_SUCCESS(
                    cudaMemcpyAsync(data.data(),
                                    allocation->ptr(),
                                    data.size(),
                                    cudaMemcpyDeviceToHost,
                                    streams[worker]));
                PADDLE_ENFORCE_GPU_SUCCESS(
                    cudaStreamSynchronize(streams[worker]));
                for (auto byte : data) {
                  if (byte != value) {
                    throw std::runtime_error(
                        "Active VMM allocation was corrupted");
                  }
                }
                // Free with work still queued to exercise remap safety events.
                PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
                    allocation->ptr(), value, data.size(), streams[worker]));
              }
            }
            ++completed[worker];
          }
          if (worker < streams.size()) {
            PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(streams[worker]));
          }
        } catch (...) {
          errors[worker] = std::current_exception();
        }
      });
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
      thread.join();
    }
    for (size_t worker = 0; worker < threads.size(); ++worker) {
      SCOPED_TRACE(worker);
      if (errors[worker]) {
        try {
          std::rethrow_exception(errors[worker]);
        } catch (const std::exception& ex) {
          ADD_FAILURE() << ex.what();
        } catch (...) {
          ADD_FAILURE() << "Unexpected worker exception";
        }
      }
      EXPECT_EQ(completed[worker], kIterations);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_NO_THROW(allocators[0]->Release(place));
    EXPECT_TRUE(multi->small_allocator()->all_blocks().empty());
    EXPECT_TRUE(multi->large_allocator()->all_blocks().empty());
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
}

using DiagnosticBlocks =
    std::vector<std::vector<std::tuple<size_t, uintptr_t, bool>>>;

TEST(VMMV2OOMDiagnostics, MessageSectionHeaderFitsTitle) {
  // Short titles keep the 22-dash width of "Error Message Summary:".
  EXPECT_EQ(MessageSectionHeader("Short"),
            "\n----------------------\nShort\n----------------------\n");
  const std::string title = "Memory Status SnapShot (GPU 12)";
  const std::string line(title.size(), '-');
  EXPECT_EQ(MessageSectionHeader(title),
            "\n" + line + "\n" + title + "\n" + line + "\n");
}

TEST(VMMV2OOMDiagnostics, AllBlockInfoSummaryCountsBlocks) {
  EXPECT_EQ(AllBlockInfoSummary({}, {}),
            "all_block_info summary: groups=0 (no mapped pool), blocks=0, "
            "mapped=0 B, used=0 B, free=0 B, free_blocks=0, max_free=0 B\n");
  const DiagnosticBlocks blocks = {
      {{512, 0x1000, false}, {1536, 0x1200, true}},
      {{3UL << 20, 0x200000, false}, {1UL << 20, 0x500000, true}}};
  EXPECT_EQ(
      AllBlockInfoSummary(blocks, {"group 0=small pool", "group 1=large pool"}),
      "all_block_info summary: groups=2 (group 0=small pool, group "
      "1=large pool), blocks=4, mapped=4.00 MB, used=3.00 MB, "
      "free=1.00 MB, free_blocks=2, max_free=1.00 MB\n");
}

TEST(VMMV2OOMDiagnostics, MemorySummaryTablesFormatsEveryBucket) {
  constexpr size_t kMiB = 1UL << 20;
  constexpr size_t kGiB = 1UL << 30;
  // One block per bucket boundary pair: KB, MB and GB sizes, a block of
  // exactly a threshold size (next bucket) and one in the +INF bucket.
  const DiagnosticBlocks blocks = {{{512, 1, false},
                                    {kMiB, 2, false},
                                    {10 * kMiB, 3, true},
                                    {50 * kMiB, 4, false},
                                    {100 * kMiB, 5, false},
                                    {200 * kMiB, 6, false},
                                    {400 * kMiB, 7, false},
                                    {600 * kMiB, 8, false},
                                    {800 * kMiB, 9, false},
                                    {kGiB, 10, false},
                                    {2 * kGiB, 11, false},
                                    {3 * kGiB, 12, false},
                                    {4 * kGiB, 13, true}}};
  const auto tables = RemoveWhitespace(MemorySummaryTables(
      blocks, phi::GPUPlace(0), -1, 0, 1UL << 30, 2UL << 30));
  // Negative stats are clamped, zero reserved gives a 0% fragmentation rate.
  EXPECT_NE(tables.find("|1|0B|"), std::string::npos) << tables;
  EXPECT_NE(tables.find("|0B|"), std::string::npos) << tables;
  EXPECT_NE(tables.find("|0.00%|1.00GB/2.00GB|"), std::string::npos) << tables;
  EXPECT_NE(tables.find("|Allocator_0|13|11|2|"), std::string::npos) << tables;
  EXPECT_NE(tables.find("|3.00GB|4.00GB|"), std::string::npos) << tables;
  EXPECT_NE(tables.find("|Allocator_0|FreeBlocks|0|0|1|0|0|0|0|0|0|0|0|1|"),
            std::string::npos)
      << tables;
  EXPECT_NE(tables.find("|Allocator_0|UsedBlocks|1|1|0|1|1|1|1|1|1|1|1|1|"),
            std::string::npos)
      << tables;

  const auto rate = RemoveWhitespace(
      MemorySummaryTables({}, phi::GPUPlace(0), 2048, 4L << 20, 0, 0));
  EXPECT_EQ(rate.find("Allocator_0"), std::string::npos) << rate;
  EXPECT_NE(rate.find("|0|2.00KB|"), std::string::npos) << rate;
  EXPECT_NE(rate.find("|4.00MB|"), std::string::npos) << rate;
}

TEST(VMMV2OOMDiagnostics, DumpAllBlockInfoToFileWritesUniqueFiles) {
  ScopedTemporaryCwd cwd;
  const DiagnosticBlocks blocks = {{{4096, 1024, false}, {8192, 5120, true}}};
  const auto first = DumpAllBlockInfoToFile(blocks, 3);
  const auto second = DumpAllBlockInfoToFile({}, 3);
  const auto first_path = SavedDumpPath(first);
  const auto second_path = SavedDumpPath(second);
  ASSERT_FALSE(first_path.empty()) << first;
  ASSERT_FALSE(second_path.empty()) << second;
  EXPECT_NE(first_path, second_path);
  EXPECT_NE(first_path.filename().string().find("_gpu3_pid"),
            std::string::npos);
  EXPECT_EQ(ReadTextFile(first_path),
            "[\n  [\n    (4096, 1024, False),\n    (8192, 5120, True),\n  ],\n"
            "]\n");
  EXPECT_EQ(ReadTextFile(second_path), "[\n]\n");
}

#ifndef _WIN32
// Makes every write to a regular file fail with EFBIG.
class ScopedZeroFileSizeLimit {
 public:
  ScopedZeroFileSizeLimit() {
    previous_handler_ = signal(SIGXFSZ, SIG_IGN);
    getrlimit(RLIMIT_FSIZE, &previous_limit_);
    rlimit limit = previous_limit_;
    limit.rlim_cur = 0;
    setrlimit(RLIMIT_FSIZE, &limit);
  }
  ~ScopedZeroFileSizeLimit() {
    setrlimit(RLIMIT_FSIZE, &previous_limit_);
    signal(SIGXFSZ, previous_handler_);
  }

 private:
  rlimit previous_limit_{};
  sighandler_t previous_handler_{nullptr};
};

TEST(VMMV2OOMDiagnostics, DumpAllBlockInfoToFileReportsWriteFailures) {
  ScopedTemporaryCwd cwd;
  // A small dump stays in the stdio buffer and fails on fclose; a dump larger
  // than the buffer fails while writing and is detected by ferror. No output
  // is produced until the limit is restored.
  DiagnosticBlocks large(1);
  for (size_t i = 0; i < 4096; ++i) {
    large[0].emplace_back(1UL << 20, i << 20, i % 2 == 0);
  }
  std::string small_result;
  std::string large_result;
  {
    ScopedZeroFileSizeLimit limit;
    small_result = DumpAllBlockInfoToFile({{{4096, 1024, false}}}, 0);
    large_result = DumpAllBlockInfoToFile(large, 0);
  }
  for (const auto& result : {small_result, large_result}) {
    EXPECT_EQ(result.rfind("Failed to write all_block_info to ", 0), 0UL)
        << result;
    EXPECT_NE(result.find("File too large"), std::string::npos) << result;
  }
  // Partially written dumps are removed.
  EXPECT_TRUE(std::filesystem::is_empty("."));
}
#endif

TEST(VMMV2OOMDiagnostics, BuildOOMMemoryDiagnosticsSkipsMissingPools) {
  ScopedTemporaryCwd cwd;
  // A null small pool is skipped; the large pool forms group 0.
  VMMAutoGrowthBestFitMultiPoolAllocatorV2 missing_small(
      nullptr,
      CreatePoolAllocator(2UL << 20, PoolType::kLarge),
      2UL << 20,
      phi::GPUPlace());
  auto allocation = missing_small.large_allocator()->Allocate(2UL << 20);
  const auto message = BuildOOMMemoryDiagnostics(&missing_small,
                                                 phi::GPUPlace(0),
                                                 2L << 20,
                                                 2L << 20,
                                                 1UL << 30,
                                                 2UL << 30);
  EXPECT_EQ(
      message.rfind(MessageSectionHeader("Memory Status SnapShot (GPU 0)"), 0),
      0UL)
      << message;
  EXPECT_NE(message.find("all_block_info summary: groups=1 (group 0=large "
                         "pool), blocks=1, mapped=2.00 MB"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("### Block Size Distribution"), std::string::npos);
  ExpectBlockInfoDump(message,
                      {missing_small.large_allocator()->SnapshotBlockInfo()});
}

TEST(VMMV2OOMDiagnostics, BuildOOMMemoryDiagnosticsFallsBackOnError) {
  ScopedTemporaryCwd cwd;
  auto allocator = CreateAllocator();
  // Only the small pool is populated, so the empty large pool is skipped.
  auto allocation = allocator->small_allocator()->Allocate(4096);
  // Device statistics are only registered for 16 devices, so building the
  // tables fails; the caller still gets a one-line notice instead of a throw.
  EXPECT_EQ(
      BuildOOMMemoryDiagnostics(allocator.get(), phi::GPUPlace(16), 0, 0, 0, 0),
      "all_block_info / memory_summary diagnostics unavailable.\n");
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
