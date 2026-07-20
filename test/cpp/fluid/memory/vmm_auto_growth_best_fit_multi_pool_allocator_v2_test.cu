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

#include <cstdlib>
#include <iostream>
#include <string>

#include "gtest/gtest.h"

#include "paddle/common/flags.h"
#include "paddle/phi/core/memory/allocation/cuda_allocator.h"
#include "paddle/phi/core/memory/allocation/retry_allocator.h"
#include "paddle/phi/core/memory/allocation/stat_allocator.h"
#include "paddle/phi/core/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

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
                                     size_t created_handles = 1)
      : VMMAutoGrowthBestFitMultiPoolAllocatorV2(
            CreatePoolAllocator(2UL << 20, PoolType::kSmall),
            CreatePoolAllocator(2UL << 20, PoolType::kLarge),
            2UL << 20,
            phi::GPUPlace()),
        requested_handles_(requested_handles),
        created_handles_(created_handles) {}

  size_t allocation_attempts() const { return allocation_attempts_; }

 protected:
  phi::Allocation* AllocateImpl(size_t) override {
    ++allocation_attempts_;
    throw VMMGrowOOM("deterministic VMM stream-safe allocation failure",
                     __FILE__,
                     __LINE__,
                     VMMGrowOOMInfo{/*requested_handles=*/requested_handles_,
                                    /*created_handles=*/created_handles_,
                                    /*handle_size=*/2UL << 20,
                                    /*device=*/0,
                                    PoolType::kLarge});
  }

 private:
  size_t allocation_attempts_{0};
  size_t requested_handles_;
  size_t created_handles_;
};

class RetryWithoutRemapMultiPoolAllocator
    : public VMMAutoGrowthBestFitMultiPoolAllocatorV2 {
 public:
  explicit RetryWithoutRemapMultiPoolAllocator(bool third_attempt_succeeds)
      : VMMAutoGrowthBestFitMultiPoolAllocatorV2(
            CreatePoolAllocator(2UL << 20, PoolType::kSmall),
            CreatePoolAllocator(2UL << 20, PoolType::kLarge),
            2UL << 20,
            phi::GPUPlace()),
        third_attempt_succeeds_(third_attempt_succeeds) {}

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
    if (allocation_attempts_ <= 2 || !third_attempt_succeeds_) {
      throw VMMGrowOOM("deterministic concurrent-free allocation failure",
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
  bool third_attempt_succeeds_;
};

void PrepareRemapSource(FailingMultiPoolAllocator* allocator,
                        AllocationPtr* anchor) {
  auto source = allocator->large_allocator()->Allocate(2UL << 20);
  *anchor = allocator->large_allocator()->Allocate(2UL << 20);
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

  // Compact is only enabled for the large pool. Small requests keep using
  // normal mapped-free reuse/grow without remap overhead.
  EXPECT_EQ(allocator->RemapForAllocation(phi::GPUPlace(), 256UL), 0UL);
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

  // Unbounded compact is an explicit maintenance path, but still only for the
  // large pool.
  EXPECT_EQ(second_allocator->RemapForAllocation(phi::GPUPlace(), 0),
            2UL << 20);
  EXPECT_EQ(second_allocator->small_allocator()->all_blocks().front().type_,
            BlockType::kFree);
  EXPECT_EQ(second_allocator->large_allocator()->all_blocks().front().type_,
            BlockType::kUnmappedFree);
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
  const phi::GPUPlace place(0);
  constexpr size_t kImpossibleSize = 1ULL << 50;

  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>();
    AllocationPtr anchor;
    PrepareRemapSource(multi.get(), &anchor);
    auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
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
      EXPECT_NE(message.find("Allocation failure summary:"), std::string::npos);
      EXPECT_EQ(message.find("Please stop other processes"), std::string::npos);
      EXPECT_NE(message.find("2. Retry after reclaiming pending frees:"),
                std::string::npos);
      EXPECT_NE(message.find("3. Retry after memory defragmentation:"),
                std::string::npos);
    }
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
  }
  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>();
    auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
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
    EXPECT_EQ(multi->allocation_attempts(), 2UL);
  }
  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>(
        /*requested_handles=*/2, /*created_handles=*/0);
    AllocationPtr anchor;
    PrepareRemapSource(multi.get(), &anchor);
    auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
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
    EXPECT_EQ(multi->allocation_attempts(), 2UL);
  }
  {
    auto multi = std::make_shared<FailingMultiPoolAllocator>();
    auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
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
    EXPECT_EQ(multi->allocation_attempts(), 2UL);
  }
  {
    auto cuda_allocator = std::make_shared<CUDAAllocator>(place);
    auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
        cuda_allocator, place, cudaStreamPerThread);
    EXPECT_THROW(stream_safe->Allocate(kImpossibleSize), BadAlloc);
  }
}

TEST(VMMAutoGrowthBestFitMultiPoolAllocatorV2,
     StreamSafeRetriesWhenFreeBlockAppearsBeforeRemap) {
  const phi::GPUPlace place(0);
  constexpr size_t kRequestSize = 4UL << 20;
  ScopedVMMRetryFlags flags(/*retry_times=*/0, /*remap_on_oom=*/true);

  {
    auto multi = std::make_shared<RetryWithoutRemapMultiPoolAllocator>(true);
    multi->PrepareRetryBlock(kRequestSize);
    auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);

    auto allocation = stream_safe->Allocate(kRequestSize);
    ASSERT_NE(allocation, nullptr);
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
  }

  {
    auto multi = std::make_shared<RetryWithoutRemapMultiPoolAllocator>(false);
    multi->PrepareRetryBlock(kRequestSize);
    auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
        multi, place, cudaStreamPerThread);

    try {
      stream_safe->Allocate(kRequestSize);
      FAIL() << "Expected the third allocation attempt to fail";
    } catch (const BadAlloc& ex) {
      const std::string message = ex.what();
      EXPECT_NE(message.find("remap was skipped because a sufficiently large "
                             "free block was available"),
                std::string::npos);
      EXPECT_NE(
          message.find("3. Retry after detecting an available free block:"),
          std::string::npos);
      EXPECT_EQ(message.find("remap moved 0.000000B"), std::string::npos);
    }
    EXPECT_EQ(multi->allocation_attempts(), 3UL);
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
