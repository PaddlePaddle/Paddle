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

#include "paddle/fluid/memory/allocation/allocator.h"
#include <map>
#include <vector>
#include "paddle/fluid/memory/allocation/aligned_allocator.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/allocation/auto_increment_allocator.h"
#include "paddle/fluid/memory/allocation/best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/locked_allocator.h"
#include "paddle/fluid/memory/allocation/naive_managed_allocator.h"
#include "paddle/fluid/memory/allocation/pinned_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

// TODO(yy): Dirty code here. This class should be configurable in runtime.
class CPUManagedAllocator : public ManagedAllocator {
 public:
  CPUManagedAllocator()
      : normal_allocator_(NaiveManagedAllocator::Create(
            std::unique_ptr<Allocator>(new CPUAllocator()))),
        communication_allocator_(NaiveManagedAllocator::Create(
            std::unique_ptr<Allocator>(new CPUPinnedAllocator()))) {}

  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override {
    if (attr == kCommunication) {
      return communication_allocator_->Allocate(size, attr);
    } else {
      return normal_allocator_->Allocate(size, attr);
    }
  }

  std::shared_ptr<Allocation> AllocateShared(size_t size, Attr attr) override {
    if (attr == kCommunication) {
      return communication_allocator_->AllocateShared(size, attr);
    } else {
      return normal_allocator_->AllocateShared(size, attr);
    }
  }
  bool IsAllocThreadSafe() const override { return true; }

 private:
  std::shared_ptr<ManagedAllocator> normal_allocator_;
  std::shared_ptr<ManagedAllocator> communication_allocator_;
};

#ifdef PADDLE_WITH_CUDA
// TODO(yy): Dirty code here. This class should be configurable in runtime.
class CUDAManagedAllocator : public ManagedAllocator {
 public:
  explicit CUDAManagedAllocator(int dev_id) {
    platform::CUDADeviceGuard guard(dev_id);
    max_chunk_size_ = platform::GpuMaxChunkSize();
    raw_allocator_ = NaiveManagedAllocator::Create(std::unique_ptr<Allocator>(
        new CUDAAllocator(platform::CUDAPlace(dev_id))));
    default_allocator_ = std::make_shared<AutoIncrementAllocator>(
        [this] { return std::move(BestFitAllocatorCreator()); });
  }

  ~CUDAManagedAllocator() {
    // Specify destruct order.
    default_allocator_.reset();
    chunks_.clear();
    raw_allocator_.reset();
  }

  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override {
    return default_allocator_->Allocate(size, attr);
  }
  std::shared_ptr<Allocation> AllocateShared(size_t size, Attr attr) override {
    return default_allocator_->AllocateShared(size, attr);
  }

  std::shared_ptr<ManagedAllocator> BestFitAllocatorCreator() {
    chunks_.emplace_back(raw_allocator_->Allocate(max_chunk_size_));
    auto* allocation = chunks_.back().get();
    return std::make_shared<AlignedAllocator<64u>>(
        NaiveManagedAllocator::Create(
            std::unique_ptr<Allocator>(new BestFitAllocator(allocation))));
  }
  bool IsAllocThreadSafe() const override { return true; }

 private:
  size_t max_chunk_size_;
  std::vector<std::unique_ptr<Allocation>> chunks_;
  std::shared_ptr<ManagedAllocator> raw_allocator_;
  std::shared_ptr<ManagedAllocator> default_allocator_;
};
#endif

class AllocatorFacadePrivate {
 public:
  std::map<platform::Place, std::shared_ptr<ManagedAllocator>> allocators_;

  ~AllocatorFacadePrivate() = default;

  AllocatorFacadePrivate() {
    InitCPUAllocator();
    InitCUDAAllocator();
  }

 private:
  void InitCPUAllocator() {
    allocators_[platform::CPUPlace()] = std::make_shared<CPUManagedAllocator>();
  }

  void InitCUDAAllocator() {
#ifdef PADDLE_WITH_CUDA
    for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount(); ++dev_id) {
      allocators_[platform::CUDAPlace(dev_id)] =
          std::make_shared<CUDAManagedAllocator>(dev_id);
    }
#endif
  }
};

// Pimpl. Make interface clean.
AllocatorFacade::AllocatorFacade() : m_(new AllocatorFacadePrivate()) {}
AllocatorFacade::~AllocatorFacade() { delete m_; }

AllocatorFacade& AllocatorFacade::Instance() {
  static AllocatorFacade instance;
  return instance;
}

std::shared_ptr<Allocation> AllocatorFacade::AllocShared(
    const platform::Place& place, size_t size, Allocator::Attr attr) {
  return m_->allocators_[place]->AllocateShared(size, attr);
}

std::unique_ptr<Allocation> AllocatorFacade::Alloc(const platform::Place& place,
                                                   size_t size,
                                                   Allocator::Attr attr) {
  return m_->allocators_[place]->Allocate(size, attr);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
