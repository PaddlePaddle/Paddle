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
#include "paddle/fluid/memory/allocation/best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/locked_allocator.h"
#include "paddle/fluid/memory/allocation/naive_managed_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

class AllocatorFacadePrivate {
 public:
  std::map<platform::Place, std::shared_ptr<ManagedAllocator>> allocators_;
  std::vector<std::unique_ptr<Allocation>> pre_allocations_;
  std::vector<std::shared_ptr<Allocator>> holding_allocators_;

  ~AllocatorFacadePrivate() {
    // Specify destruct order.
    pre_allocations_.clear();
    allocators_.clear();
    holding_allocators_.clear();
  }

  AllocatorFacadePrivate() {
    std::cout << "Init Allocator Facade" << std::endl;
    InitCPUAllocator();
    InitCUDAAllocator();
  }

 private:
  void InitCPUAllocator() {
    auto all = NaiveManagedAllocator::Create(
        std::unique_ptr<Allocator>(new CPUAllocator()));

    allocators_[platform::CPUPlace()] = all;
  }

  void InitCUDAAllocator() {
#ifdef PADDLE_WITH_CUDA
    for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount(); ++dev_id) {
      platform::CUDADeviceGuard guard(dev_id);
      auto cuda_allocator =
          NaiveManagedAllocator::Create(std::unique_ptr<Allocator>(
              new CUDAAllocator(platform::CUDAPlace(dev_id))));
      auto allocation = cuda_allocator->Allocate(platform::GpuMaxChunkSize());
      auto allocator = NaiveManagedAllocator::Create(std::unique_ptr<Allocator>(
          new LockedAllocator(std::unique_ptr<Allocator>(
              new BestFitAllocator(allocation.get())))));

      pre_allocations_.emplace_back(std::move(allocation));
      holding_allocators_.emplace_back(cuda_allocator);
      allocators_[platform::CUDAPlace(dev_id)] =
          std::make_shared<AlignedAllocator<64>>(std::move(allocator));
    }
#endif
  }
};

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
