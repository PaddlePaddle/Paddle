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

#include <iostream>
#include <memory>
#include <string>

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator.h"

// Expose internals for white-box testing.
#define private public
#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#undef private

#include "gtest/gtest.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/memory/memory.h"

namespace paddle {
namespace memory {
namespace allocation {

class TestCUDAVirtualMemAllocator : public CUDAVirtualMemAllocator {
 public:
  using CUDAVirtualMemAllocator::CUDAVirtualMemAllocator;
  using CUDAVirtualMemAllocator::FreeImpl;
};

TEST(test_vmm_allocator, test_mem_stats) {
  size_t alignment = 256;
  auto underlying_allocator =
      std::make_shared<TestCUDAVirtualMemAllocator>(phi::GPUPlace());
  auto allocation = underlying_allocator->Allocate(1024);
  EXPECT_GT(DeviceMemoryStatCurrentValue("Reserved", 0), 1024);
  allocation.reset();
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), 0);
}

class DummyAllocator : public Allocator {
 public:
  bool IsAllocThreadSafe() const override { return true; }

 protected:
  phi::Allocation* AllocateImpl(size_t) override {
    PADDLE_THROW(common::errors::Unavailable(
        "DummyAllocator::AllocateImpl should not be called."));
  }
  void FreeImpl(phi::Allocation*) override {}
};

class AlwaysOOMAllocator : public Allocator {
 public:
  bool IsAllocThreadSafe() const override { return true; }

 protected:
  phi::Allocation* AllocateImpl(size_t size) override {
    PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
        "AlwaysOOMAllocator failed to allocate %zu bytes.", size));
  }
  void FreeImpl(phi::Allocation*) override {}
};

// Expose FreeImpl for testing.
class ExposedVmmAllocator : public VirtualMemoryAutoGrowthBestFitAllocator {
 public:
  using VirtualMemoryAutoGrowthBestFitAllocator::FreeImpl;
  using VirtualMemoryAutoGrowthBestFitAllocator::
      VirtualMemoryAutoGrowthBestFitAllocator;
};

TEST(test_vmm_allocator, free_impl_uses_allocation_block_iterator) {
  auto underlying = std::make_shared<DummyAllocator>();
  phi::GPUPlace place(0);
  ExposedVmmAllocator allocator(underlying, 256, place);

  // Manually construct blocks: [free-prev][used-target][free-next]
  allocator.all_blocks_.clear();
  auto prev = allocator.all_blocks_.emplace(
      allocator.all_blocks_.end(), reinterpret_cast<void*>(0x1000), 1024, true);
  auto target = allocator.all_blocks_.emplace(allocator.all_blocks_.end(),
                                              reinterpret_cast<void*>(0x1400),
                                              2048,
                                              false);
  auto next = allocator.all_blocks_.emplace(
      allocator.all_blocks_.end(), reinterpret_cast<void*>(0x1C00), 4096, true);

  allocator.free_blocks_.clear();
  allocator.free_blocks_.emplace(std::make_pair(prev->size_, prev->ptr_), prev);
  allocator.free_blocks_.emplace(std::make_pair(next->size_, next->ptr_), next);

  auto allocation = std::make_unique<BlockAllocation>(target, place);

  EXPECT_NO_THROW(allocator.FreeImpl(allocation.release()));
  EXPECT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.all_blocks_.front().ptr_,
            reinterpret_cast<void*>(0x1000));
  EXPECT_EQ(allocator.all_blocks_.front().size_, 7168UL);
  EXPECT_TRUE(allocator.all_blocks_.front().is_free_);
}

TEST(test_vmm_allocator, oom_error_prints_pool_stats) {
  auto underlying = std::make_shared<AlwaysOOMAllocator>();
  phi::GPUPlace place(0);
  ExposedVmmAllocator allocator(underlying, 256, place);

  auto used1 = allocator.all_blocks_.emplace(allocator.all_blocks_.end(),
                                             reinterpret_cast<void*>(0x1000),
                                             1UL << 20,
                                             false);
  auto free1 = allocator.all_blocks_.emplace(allocator.all_blocks_.end(),
                                             reinterpret_cast<void*>(0x101000),
                                             2UL << 20,
                                             true);
  auto used2 = allocator.all_blocks_.emplace(allocator.all_blocks_.end(),
                                             reinterpret_cast<void*>(0x301000),
                                             1UL << 20,
                                             false);
  auto free2 = allocator.all_blocks_.emplace(allocator.all_blocks_.end(),
                                             reinterpret_cast<void*>(0x401000),
                                             4UL << 20,
                                             true);

  allocator.free_blocks_.emplace(std::make_pair(free1->size_, free1->ptr_),
                                 free1);
  allocator.free_blocks_.emplace(std::make_pair(free2->size_, free2->ptr_),
                                 free2);

  try {
    allocator.Allocate(5UL << 20);
    FAIL() << "Expected VMM allocator OOM.";
  } catch (const BadAlloc& ex) {
    std::string message = ex.what();
    std::cout << "\n[VMM OOM MESSAGE]\n" << message << std::endl;
    EXPECT_NE(message.find("VMM allocator stats (pool): "
                           "total_free=6.000000MB, max_free=4.000000MB."),
              std::string::npos);
  }

  static_cast<void>(used1);
  static_cast<void>(used2);
}

// ---------------------------------------------------------------------------
// Change B/C: ReleaseImpl integration tests on real device memory. These run
// before the synthetic IPC-registry tests below so a stray synthetic mark can
// never make a real chunk look IPC-exported.
// ---------------------------------------------------------------------------
TEST(test_vmm_allocator, release_impl_releases_free_and_keeps_active_chunk) {
  const size_t alignment = 256;
  phi::GPUPlace place(0);
  auto underlying = std::make_shared<TestCUDAVirtualMemAllocator>(place);
  ExposedVmmAllocator allocator(underlying, alignment, place);

  const size_t before = DeviceMemoryStatCurrentValue("Reserved", 0);
  auto big = allocator.Allocate(4UL << 20);  // one fully-active chunk
  const size_t reserved_active = DeviceMemoryStatCurrentValue("Reserved", 0);
  EXPECT_GT(reserved_active, before);

  // A chunk with a live allocation must not be released.
  EXPECT_EQ(allocator.Release(place), 0UL);
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), reserved_active);

  big.reset();  // the whole chunk becomes free
  const uint64_t released = allocator.Release(place);
  EXPECT_GT(released, 0UL);
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), before);
  EXPECT_TRUE(allocator.allocations_.empty());
}

TEST(test_vmm_allocator, multiscale_pool_release_delegates_to_sub_pools) {
  const size_t alignment = 256;
  phi::GPUPlace place(0);
  auto small = std::make_shared<VirtualMemoryAutoGrowthBestFitAllocator>(
      std::make_shared<TestCUDAVirtualMemAllocator>(place), alignment, place);
  auto large = std::make_shared<VirtualMemoryAutoGrowthBestFitAllocator>(
      std::make_shared<TestCUDAVirtualMemAllocator>(place), alignment, place);
  VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator pool(
      small, large, alignment, place);

  const size_t before = DeviceMemoryStatCurrentValue("Reserved", 0);
  auto small_alloc = pool.Allocate(256UL << 10);  // routed by pool policy
  auto large_alloc = pool.Allocate(4UL << 20);
  EXPECT_GT(DeviceMemoryStatCurrentValue("Reserved", 0), before);

  small_alloc.reset();
  large_alloc.reset();
  // Pool ReleaseImpl sums Release() over both sub-pools.
  EXPECT_GT(pool.Release(place), 0UL);
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), before);
}

// ---------------------------------------------------------------------------
// Change B: RemoveFreeRange white-box unit tests (no device memory needed).
// Blocks are built with empty parts_; SliceBlockPartsForRange short-circuits on
// empty parts, so the pure interval-splitting logic can be exercised directly.
// ---------------------------------------------------------------------------
namespace {
std::list<Block>::iterator AddFreeBlock(
    VirtualMemoryAutoGrowthBestFitAllocator* allocator,
    uintptr_t ptr,
    size_t size) {
  auto it = allocator->all_blocks_.emplace(
      allocator->all_blocks_.end(), reinterpret_cast<void*>(ptr), size, true);
  allocator->free_blocks_.emplace(std::make_pair(size, it->ptr_), it);
  return it;
}
}  // namespace

TEST(test_vmm_allocator, remove_free_range_full_cover) {
  auto underlying = std::make_shared<DummyAllocator>();
  phi::GPUPlace place(0);
  ExposedVmmAllocator allocator(underlying, 256, place);
  AddFreeBlock(&allocator, 0x1000, 0x1000);  // [0x1000, 0x2000)
  allocator.RemoveFreeRange(0x1000, 0x2000);
  EXPECT_TRUE(allocator.all_blocks_.empty());
  EXPECT_TRUE(allocator.free_blocks_.empty());
}

TEST(test_vmm_allocator, remove_free_range_keeps_left_remnant) {
  auto underlying = std::make_shared<DummyAllocator>();
  phi::GPUPlace place(0);
  ExposedVmmAllocator allocator(underlying, 256, place);
  AddFreeBlock(&allocator, 0x1000, 0x1000);   // [0x1000, 0x2000)
  allocator.RemoveFreeRange(0x1800, 0x2000);  // drop the tail 0x800
  ASSERT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.all_blocks_.front().ptr_,
            reinterpret_cast<void*>(0x1000));
  EXPECT_EQ(allocator.all_blocks_.front().size_, 0x800UL);
  EXPECT_TRUE(allocator.all_blocks_.front().is_free_);
  EXPECT_EQ(allocator.free_blocks_.count(
                std::make_pair(0x800UL, reinterpret_cast<void*>(0x1000))),
            1UL);
}

TEST(test_vmm_allocator, remove_free_range_keeps_right_remnant) {
  auto underlying = std::make_shared<DummyAllocator>();
  phi::GPUPlace place(0);
  ExposedVmmAllocator allocator(underlying, 256, place);
  AddFreeBlock(&allocator, 0x1000, 0x1000);   // [0x1000, 0x2000)
  allocator.RemoveFreeRange(0x1000, 0x1800);  // drop the head 0x800
  ASSERT_EQ(allocator.all_blocks_.size(), 1UL);
  EXPECT_EQ(allocator.all_blocks_.front().ptr_,
            reinterpret_cast<void*>(0x1800));
  EXPECT_EQ(allocator.all_blocks_.front().size_, 0x800UL);
  EXPECT_TRUE(allocator.all_blocks_.front().is_free_);
  EXPECT_EQ(allocator.free_blocks_.count(
                std::make_pair(0x800UL, reinterpret_cast<void*>(0x1800))),
            1UL);
}

TEST(test_vmm_allocator, remove_free_range_splits_into_both_remnants) {
  auto underlying = std::make_shared<DummyAllocator>();
  phi::GPUPlace place(0);
  ExposedVmmAllocator allocator(underlying, 256, place);
  AddFreeBlock(&allocator, 0x1000, 0x1000);   // [0x1000, 0x2000)
  allocator.RemoveFreeRange(0x1400, 0x1C00);  // drop the middle
  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  EXPECT_EQ(allocator.all_blocks_.front().ptr_,
            reinterpret_cast<void*>(0x1000));
  EXPECT_EQ(allocator.all_blocks_.front().size_, 0x400UL);
  EXPECT_EQ(allocator.all_blocks_.back().ptr_, reinterpret_cast<void*>(0x1C00));
  EXPECT_EQ(allocator.all_blocks_.back().size_, 0x400UL);
  EXPECT_EQ(allocator.free_blocks_.size(), 2UL);
}

TEST(test_vmm_allocator, remove_free_range_spans_multiple_blocks) {
  auto underlying = std::make_shared<DummyAllocator>();
  phi::GPUPlace place(0);
  ExposedVmmAllocator allocator(underlying, 256, place);
  AddFreeBlock(&allocator, 0x1000, 0x800);  // [0x1000, 0x1800)
  AddFreeBlock(&allocator, 0x1800, 0x800);  // [0x1800, 0x2000)
  allocator.RemoveFreeRange(0x1000, 0x2000);
  EXPECT_TRUE(allocator.all_blocks_.empty());
  EXPECT_TRUE(allocator.free_blocks_.empty());
}

TEST(test_vmm_allocator, remove_free_range_partially_spans_two_blocks) {
  auto underlying = std::make_shared<DummyAllocator>();
  phi::GPUPlace place(0);
  ExposedVmmAllocator allocator(underlying, 256, place);
  AddFreeBlock(&allocator, 0x1000, 0x800);    // [0x1000, 0x1800)
  AddFreeBlock(&allocator, 0x1800, 0x800);    // [0x1800, 0x2000)
  allocator.RemoveFreeRange(0x1400, 0x1C00);  // trims tail of #1, head of #2
  ASSERT_EQ(allocator.all_blocks_.size(), 2UL);
  EXPECT_EQ(allocator.all_blocks_.front().ptr_,
            reinterpret_cast<void*>(0x1000));
  EXPECT_EQ(allocator.all_blocks_.front().size_, 0x400UL);
  EXPECT_EQ(allocator.all_blocks_.back().ptr_, reinterpret_cast<void*>(0x1C00));
  EXPECT_EQ(allocator.all_blocks_.back().size_, 0x400UL);
}

// ---------------------------------------------------------------------------
// Change A: CUDAVirtualMemAllocator IPC-export interval registry. The registry
// is a static/process-wide map with no reset hook, so each test uses a distinct
// synthetic address region that real GPU VMM reservations never return.
// ---------------------------------------------------------------------------
TEST(test_vmm_allocator, ipc_exported_range_overlap_queries) {
  const uintptr_t base = 0x40000000;  // 1 GiB synthetic region
  CUDAVirtualMemAllocator::MarkIPCExported(reinterpret_cast<void*>(base),
                                           0x1000);  // [base, base+0x1000)
  auto Q = [](uintptr_t p, size_t n) {
    return CUDAVirtualMemAllocator::AnyIPCExportedInRange(
        reinterpret_cast<void*>(p), n);
  };
  EXPECT_TRUE(Q(base + 0x400, 0x100));    // fully inside
  EXPECT_TRUE(Q(base - 0x400, 0x800));    // overlaps the left edge
  EXPECT_TRUE(Q(base, 0x100));            // starts exactly at begin
  EXPECT_TRUE(Q(base, 0x1000));           // exact match
  EXPECT_TRUE(Q(base + 0xF00, 0x400));    // overlaps the right edge
  EXPECT_FALSE(Q(base - 0x400, 0x400));   // adjacent on the left, no overlap
  EXPECT_FALSE(Q(base + 0x1000, 0x400));  // adjacent on the right, no overlap
}

TEST(test_vmm_allocator, ipc_exported_guards_null_and_zero) {
  CUDAVirtualMemAllocator::MarkIPCExported(nullptr, 0x1000);  // no-op
  CUDAVirtualMemAllocator::MarkIPCExported(reinterpret_cast<void*>(0x50000000),
                                           0);  // no-op
  EXPECT_FALSE(CUDAVirtualMemAllocator::AnyIPCExportedInRange(nullptr, 0x1000));
  EXPECT_FALSE(CUDAVirtualMemAllocator::AnyIPCExportedInRange(
      reinterpret_cast<void*>(0x50000000), 0));
  EXPECT_FALSE(CUDAVirtualMemAllocator::AnyIPCExportedInRange(
      reinterpret_cast<void*>(0x60000000), 0x1000));  // never marked
}

TEST(test_vmm_allocator, ipc_exported_extends_end_but_never_shrinks) {
  const uintptr_t base = 0x70000000;
  auto mark = [](uintptr_t p, size_t n) {
    CUDAVirtualMemAllocator::MarkIPCExported(reinterpret_cast<void*>(p), n);
  };
  auto Q = [](uintptr_t p, size_t n) {
    return CUDAVirtualMemAllocator::AnyIPCExportedInRange(
        reinterpret_cast<void*>(p), n);
  };
  mark(base, 0x800);  // [base, base+0x800)
  EXPECT_FALSE(Q(base + 0x900, 0x100));
  mark(base, 0x1000);  // extend end to base+0x1000
  EXPECT_TRUE(Q(base + 0x900, 0x100));
  mark(base, 0x400);  // a smaller size must not shrink the recorded end
  EXPECT_TRUE(Q(base + 0x900, 0x100));
}

// Must be the last device test: it pins a real chunk VA in the process-wide
// IPC-export registry (which has no reset hook), so no later ReleaseImpl test
// may reuse that VA.
TEST(test_vmm_allocator, release_impl_skips_ipc_exported_chunk) {
  const size_t alignment = 256;
  phi::GPUPlace place(0);
  auto underlying = std::make_shared<TestCUDAVirtualMemAllocator>(place);
  ExposedVmmAllocator allocator(underlying, alignment, place);

  auto a = allocator.Allocate(4UL << 20);
  ASSERT_FALSE(allocator.allocations_.empty());
  void* chunk_ptr = allocator.allocations_.front()->ptr();
  const size_t chunk_size = allocator.allocations_.front()->size();
  a.reset();  // free, then pin as IPC-exported
  const size_t reserved = DeviceMemoryStatCurrentValue("Reserved", 0);

  CUDAVirtualMemAllocator::MarkIPCExported(chunk_ptr, chunk_size);
  // Free but IPC-exported -> ReleaseImpl must keep the backing mapped.
  EXPECT_EQ(allocator.Release(place), 0UL);
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), reserved);
  EXPECT_FALSE(allocator.allocations_.empty());
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
