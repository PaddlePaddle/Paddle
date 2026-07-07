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
#include "paddle/common/flags.h"

#include "glog/logging.h"

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/core/memory/allocation/stream_safe_cuda_allocator.h"
#endif

COMMON_DECLARE_int64(offload_retry_times);
COMMON_DECLARE_bool(vmm_v2_remap_on_oom);

namespace paddle::memory::allocation {

namespace {

bool CanTryVMMV2Remap(const std::shared_ptr<Allocator>& allocator) {
#if defined(PADDLE_WITH_CUDA)
  if (!FLAGS_vmm_v2_remap_on_oom) {
    return false;
  }
  auto* stream_safe_allocator =
      dynamic_cast<StreamSafeCUDAAllocator*>(allocator.get());
  return stream_safe_allocator != nullptr &&
         stream_safe_allocator->GetVMMV2Allocator() != nullptr;
#else
  (void)allocator;
  return false;
#endif
}

}  // namespace

static std::function<size_t(Place, size_t)> g_oom_callback;

void RegisterOOMCallback(std::function<size_t(Place, size_t)> callback) {
  g_oom_callback = std::move(callback);
}

class WaitedAllocateSizeGuard {
 public:
  WaitedAllocateSizeGuard(std::atomic<size_t>* waited_size,
                          size_t requested_size)
      : waited_size_(waited_size), requested_size_(requested_size) {
    waited_size_->fetch_add(requested_size_, std::memory_order_relaxed);
  }

  ~WaitedAllocateSizeGuard() {
    waited_size_->fetch_sub(requested_size_, std::memory_order_relaxed);
  }

 private:
  std::atomic<size_t>* waited_size_;
  size_t requested_size_;
};

void RetryAllocator::FreeImpl(phi::Allocation* allocation) {
  // Delete underlying allocation first.
  size_t size = allocation->size();
  underlying_allocator_->Free(allocation);
  if (UNLIKELY(waited_allocate_size_)) {
    VLOG(10) << "Free " << size
             << " bytes and notify all waited threads, "
                "where waited_allocate_size_ = "
             << waited_allocate_size_;
    cv_.notify_all();
  }
}

phi::Allocation* RetryAllocator::AllocateImpl(size_t size) {
  auto alloc_func = [&, this]() {
    return underlying_allocator_->Allocate(size).release();
  };
  auto try_remap = [&, this]() -> bool {
    if (!CanTryVMMV2Remap(underlying_allocator_)) {
      return false;
    }
    try {
      const size_t remapped = underlying_allocator_->Compact(place_, size);
      VLOG(4) << "RetryAllocator: VMM V2 compact returned " << remapped
              << " bytes";
      return remapped > 0;
    } catch (const std::exception& e) {
      VLOG(4) << "VMM V2 compact on " << place_
              << " failed with exception: " << e.what();
      return false;
    } catch (...) {
      VLOG(4) << "VMM V2 compact on " << place_
              << " failed with unknown exception.";
      return false;
    }
  };
  // In fact, we can unify the code of allocation success and failure
  // But it would add lock even when allocation success at the first time
  try {
    if (FLAGS_offload_retry_times <= 0 || g_oom_callback == nullptr) {
      return alloc_func();
    }

    bool has_offloaded = true;
    for (int64_t i = 0; i < FLAGS_offload_retry_times && has_offloaded; ++i) {
      try {
        return alloc_func();
      } catch (BadAlloc&) {
        VLOG(10) << "Allocation " << size << " on " << place_
                 << " failed, try offload on retry " << i;
        has_offloaded = (g_oom_callback(place_, size) > 0);

        if (!has_offloaded) {
          continue;
        }

        // Offload may already have created a large enough free block. Retry
        // allocation first. If it still fails, only a VMM V2-backed CUDA
        // allocator may run one extra post-offload compact; other allocator
        // stacks keep their legacy retry behavior.
        try {
          return alloc_func();
        } catch (BadAlloc&) {
        }

        if (try_remap()) {
          try {
            return alloc_func();
          } catch (BadAlloc&) {
          }
        }
      }
    }
    return alloc_func();
  } catch (BadAlloc&) {
    {
      WaitedAllocateSizeGuard guard(&waited_allocate_size_, size);
      VLOG(10) << "Allocation failed when allocating " << size
               << " bytes, waited_allocate_size_ = " << waited_allocate_size_;
      // We can just write allocation retry inside the predicate function of
      // wait_until. But it needs to acquire the lock when executing predicate
      // function. For better performance, we use loop here
      auto end_time = std::chrono::high_resolution_clock::now() + retry_time_;
      auto wait_until = [&, this] {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_until(lock, end_time);
      };

      size_t retry_time = 0;
      while (wait_until() != std::cv_status::timeout) {
        try {
          return alloc_func();
        } catch (BadAlloc&) {
          // do nothing when it is not timeout
          ++retry_time;
          VLOG(10) << "Allocation failed when retrying " << retry_time
                   << " times when allocating " << size
                   << " bytes. Wait still.";
        } catch (...) {
          throw;
        }
      }
    }
    VLOG(10) << "Allocation failed because of timeout when allocating " << size
             << " bytes.";
    return alloc_func();  // If timeout, try last allocation request.
  } catch (...) {
    throw;
  }
}

}  // namespace paddle::memory::allocation
