// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/retry_allocator.h"

#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/phi/core/memory/allocation/best_fit_allocator.h"
#include "paddle/phi/core/memory/allocation/cpu_allocator.h"
#include "paddle/phi/core/memory/allocation/stat_allocator.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/memory/allocation/cuda_allocator.h"
#endif
#if defined(PADDLE_WITH_CUDA)
#include "glog/logging.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/stream_safe_cuda_allocator.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"
#endif

COMMON_DECLARE_int64(offload_retry_times);
COMMON_DECLARE_bool(vmm_v2_remap_on_oom);

namespace paddle {
namespace memory {
namespace allocation {

TEST(RetryAllocator, RetryAllocator) {
  CPUAllocator cpu_allocator;

  size_t size = (1 << 20);
  auto cpu_allocation = cpu_allocator.Allocate(size);

  size_t thread_num = 4;
  size_t sleep_time = 40;
  size_t extra_time = 20;

  // Reserve to perform more tests in the future
  std::vector<std::shared_ptr<Allocator>> allocators;
  {
    std::unique_ptr<BestFitAllocator> best_fit_allocator(
        new BestFitAllocator(cpu_allocation.get()));
    allocators.push_back(std::make_shared<RetryAllocator>(
        std::move(best_fit_allocator),
        phi::CPUPlace(),
        (thread_num - 1) * (sleep_time + extra_time)));
  }

  for (auto &allocator : allocators) {
    std::vector<std::thread> threads(thread_num);
    std::vector<void *> addresses(threads.size(), nullptr);

    std::mutex mutex;
    std::condition_variable cv;
    bool flag = false;

    for (size_t i = 0; i < threads.size(); ++i) {
      threads[i] = std::thread([&, i]() {
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return flag; });
        }

        auto ret = allocator->Allocate(size - 1);
        addresses[i] = ret->ptr();
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
      });
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      flag = true;
      cv.notify_all();
    }

    for (auto &th : threads) {
      th.join();
    }

    void *val = cpu_allocation->ptr();
    bool is_all_equal = std::all_of(addresses.begin(),
                                    addresses.end(),
                                    [val](void *p) { return p == val; });
    ASSERT_TRUE(is_all_equal);
    allocator->Release(phi::CPUPlace());
  }
}

class DummyAllocator : public Allocator {
 public:
  bool IsAllocThreadSafe() const override { return true; }

 protected:
  phi::Allocation *AllocateImpl(size_t size) override {
    PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
        "Here is a test exception, always BadAlloc."));
  }

  void FreeImpl(phi::Allocation *) override {}
};

class OffloadRetryNoCompactAllocator : public Allocator {
 public:
  explicit OffloadRetryNoCompactAllocator(size_t storage_size)
      : storage_(storage_size) {}

  bool IsAllocThreadSafe() const override { return true; }

  size_t allocate_count() const { return allocate_count_; }
  size_t compact_count() const { return compact_count_; }

 protected:
  phi::Allocation *AllocateImpl(size_t size) override {
    ++allocate_count_;
    if (allocate_count_ < 3) {
      PADDLE_THROW_BAD_ALLOC(common::errors::ResourceExhausted(
          "test allocator fails before offload retry"));
    }
    PADDLE_ENFORCE_LE(
        size,
        storage_.size(),
        common::errors::InvalidArgument("requested test allocation too large"));
    return std::make_unique<Allocation>(storage_.data(), size, place_)
        .release();
  }

  void FreeImpl(phi::Allocation *allocation) override { delete allocation; }

  size_t CompactImpl(const Place &place) override {
    (void)place;
    ++compact_count_;
    ADD_FAILURE() << "non-VMM allocator CompactImpl should not be called";
    return 0;
  }

 private:
  size_t allocate_count_{0};
  size_t compact_count_{0};
  std::vector<char> storage_;
  phi::CPUPlace place_;
};

class ScopedRetryAllocatorFlags {
 public:
  ScopedRetryAllocatorFlags(int64_t retry_times, bool remap_on_oom)
      : old_retry_times_(FLAGS_offload_retry_times),
        old_remap_on_oom_(FLAGS_vmm_v2_remap_on_oom) {
    FLAGS_offload_retry_times = retry_times;
    FLAGS_vmm_v2_remap_on_oom = remap_on_oom;
  }

  ~ScopedRetryAllocatorFlags() {
    RegisterOOMCallback(nullptr);
    FLAGS_offload_retry_times = old_retry_times_;
    FLAGS_vmm_v2_remap_on_oom = old_remap_on_oom_;
  }

 private:
  int64_t old_retry_times_;
  bool old_remap_on_oom_;
};

#if defined(PADDLE_WITH_CUDA)
class ScopedRetryVLogLevel {
 public:
  explicit ScopedRetryVLogLevel(int level) : old_level_(FLAGS_v) {
    FLAGS_v = level;
  }
  ~ScopedRetryVLogLevel() { FLAGS_v = old_level_; }

 private:
  int old_level_;
};

std::shared_ptr<VMMAutoGrowthBestFitMultiPoolAllocatorV2>
CreateRetryTestVMMAllocator(const phi::GPUPlace &place) {
  auto small_vmm = std::make_shared<CUDAVirtualMemAllocatorV2>(
      place, 2UL << 20, PoolType::kSmall);
  auto large_vmm = std::make_shared<CUDAVirtualMemAllocatorV2>(
      place, 16UL << 20, PoolType::kLarge);
  auto small = std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
      small_vmm, 256, place, PoolType::kSmall);
  auto large = std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
      large_vmm, 256, place, PoolType::kLarge);
  return std::make_shared<VMMAutoGrowthBestFitMultiPoolAllocatorV2>(
      small, large, 2UL << 20, place);
}
#endif

TEST(RetryAllocator, OffloadRetryDoesNotCompactNonVMMAllocator) {
  auto raw_allocator = std::make_shared<OffloadRetryNoCompactAllocator>(256);
#if defined(PADDLE_WITH_CUDA)
  EXPECT_EQ(dynamic_cast<StreamSafeCUDAAllocator *>(raw_allocator.get()),
            nullptr);
#endif
  RetryAllocator allocator(raw_allocator, phi::CPUPlace(), 10);

  ScopedRetryAllocatorFlags flags_guard(/*retry_times=*/1,
                                        /*remap_on_oom=*/true);
  RegisterOOMCallback([](Place place, size_t size) -> size_t {
    EXPECT_TRUE(phi::is_cpu_place(place));
    EXPECT_EQ(size, 256UL);
    return size;
  });

  auto allocation = allocator.Allocate(256);
  EXPECT_NE(allocation->ptr(), nullptr);
  allocation.reset();

  EXPECT_EQ(raw_allocator->allocate_count(), 3UL);
  EXPECT_EQ(raw_allocator->compact_count(), 0UL);
}

#if defined(PADDLE_WITH_CUDA)
TEST(RetryAllocator, OffloadRetryDispatchesVMMCompact) {
  ScopedRetryVLogLevel vlog_guard(4);
  const phi::GPUPlace place(0);
  auto multi = CreateRetryTestVMMAllocator(place);
  auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
      multi, place, cudaStreamPerThread);
  RetryAllocator allocator(stream_safe, place, 0);

  ScopedRetryAllocatorFlags flags_guard(/*retry_times=*/1,
                                        /*remap_on_oom=*/true);
  RegisterOOMCallback([](Place, size_t size) { return size; });

  EXPECT_THROW(allocator.Allocate(1ULL << 50), BadAlloc);
}

TEST(RetryAllocator, StreamSafeVMMOOMWithoutRemap) {
  ScopedRetryVLogLevel vlog_guard(4);
  const phi::GPUPlace place(0);
  auto multi = CreateRetryTestVMMAllocator(place);
  auto stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
      multi, place, cudaStreamPerThread);
  ScopedRetryAllocatorFlags flags_guard(/*retry_times=*/0,
                                        /*remap_on_oom=*/false);

  EXPECT_THROW(stream_safe->Allocate(1ULL << 50), BadAlloc);
}

TEST(RetryAllocator, StreamSafeFindsWrappedVMMAllocator) {
  const phi::GPUPlace place(0);
  auto multi = CreateRetryTestVMMAllocator(place);
  auto retry = std::make_shared<RetryAllocator>(multi, place, 0);
  auto retry_stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
      retry, place, cudaStreamPerThread);
  EXPECT_EQ(retry_stream_safe->GetVMMV2Allocator(), multi.get());

  auto stat = std::make_shared<StatAllocator>(multi);
  auto stat_stream_safe = std::make_shared<StreamSafeCUDAAllocator>(
      stat, place, cudaStreamPerThread);
  EXPECT_EQ(stat_stream_safe->GetVMMV2Allocator(), multi.get());
}
#endif

TEST(RetryAllocator, RetryAllocatorLastAllocFailure) {
  size_t retry_ms = 10;
  {
    RetryAllocator allocator(
        std::make_shared<DummyAllocator>(), phi::CPUPlace(), retry_ms);
    try {
      auto allocation = allocator.Allocate(100);
      ASSERT_TRUE(false);
      allocation.reset();
    } catch (BadAlloc &ex) {
      ASSERT_TRUE(std::string(ex.what()).find("always BadAlloc") !=
                  std::string::npos);
    }
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    phi::GPUPlace p(0);
    RetryAllocator allocator(std::make_shared<CUDAAllocator>(p), p, retry_ms);
    size_t allocate_size = (static_cast<size_t>(1) << 40);  // Very large number
    try {
      auto allocation = allocator.Allocate(allocate_size);
      ASSERT_TRUE(false);
      allocation.reset();
      allocator.Release(p);
    } catch (BadAlloc &ex) {
      ASSERT_TRUE(std::string(ex.what()).find("Cannot allocate") !=
                  std::string::npos);
    }
  }
#endif
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
