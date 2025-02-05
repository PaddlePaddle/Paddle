// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/thread_local_allocator.h"

namespace paddle::memory::allocation {

ThreadLocalAllocatorImpl::ThreadLocalAllocatorImpl(const phi::Place& p)
    : place_(p) {
  if (phi::is_gpu_place(place_)) {
    buddy_allocator_ = std::make_unique<memory::detail::BuddyAllocator>(
        std::unique_ptr<memory::detail::SystemAllocator>(
            new memory::detail::GPUAllocator(place_.device)),
        platform::GpuMinChunkSize(),
        platform::GpuMaxChunkSize());
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "Thread local allocator only supports CUDAPlace now."));
  }
}

std::shared_ptr<ThreadLocalAllocatorImpl> ThreadLocalCUDAAllocatorPool::Get(
    int gpu_id) {
  auto pos = std::distance(devices_.begin(),
                           std::find(devices_.begin(), devices_.end(), gpu_id));
  PADDLE_ENFORCE_LT(
      pos,
      devices_.size(),
      common::errors::InvalidArgument(
          "The position of device should be less than the size of devices."));
  std::call_once(*init_flags_[pos], [this, pos, gpu_id] {
    platform::SetDeviceId(devices_[pos]);
    allocators_[pos].reset(new ThreadLocalAllocatorImpl(phi::GPUPlace(gpu_id)));
  });
  return allocators_[pos];
}

ThreadLocalCUDAAllocatorPool::ThreadLocalCUDAAllocatorPool()
    : devices_(platform::GetSelectedDevices()) {
  auto gpu_num = devices_.size();
  allocators_.resize(gpu_num);
  init_flags_.reserve(gpu_num);
  for (size_t i = 0; i < gpu_num; ++i) {
    init_flags_.emplace_back(new std::once_flag());
  }
}

ThreadLocalAllocation* ThreadLocalAllocatorImpl::AllocateImpl(size_t size) {
  VLOG(10) << "ThreadLocalAllocatorImpl::AllocateImpl " << size;
  void* ptr = buddy_allocator_->Alloc(size);
  auto* tl_allocation = new ThreadLocalAllocation(ptr, size, place_);
  tl_allocation->SetThreadLocalAllocatorImpl(shared_from_this());
  return tl_allocation;
}

void ThreadLocalAllocatorImpl::FreeImpl(ThreadLocalAllocation* allocation) {
  VLOG(10) << "ThreadLocalAllocatorImpl::FreeImpl " << allocation;
  buddy_allocator_->Free(allocation->ptr());
  delete allocation;
}

uint64_t ThreadLocalAllocatorImpl::ReleaseImpl() {
  return buddy_allocator_->Release();
}

}  // namespace paddle::memory::allocation
