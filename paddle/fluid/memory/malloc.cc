/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/malloc.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/stream.h"

namespace paddle {
namespace memory {

std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                        size_t size) {
  return allocation::AllocatorFacade::Instance().AllocShared(place, size);
}

AllocationPtr Alloc(const platform::Place& place, size_t size) {
  return allocation::AllocatorFacade::Instance().Alloc(place, size);
}

uint64_t Release(const platform::Place& place) {
  return allocation::AllocatorFacade::Instance().Release(place);
}

std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                        size_t size,
                                        const phi::Stream& stream) {
  return allocation::AllocatorFacade::Instance().AllocShared(place, size,
                                                             stream);
}

AllocationPtr Alloc(const platform::CUDAPlace& place, size_t size,
                    const phi::Stream& stream) {
  return allocation::AllocatorFacade::Instance().Alloc(place, size, stream);
}

bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                  const phi::Stream& stream) {
  return allocation::AllocatorFacade::Instance().InSameStream(allocation,
                                                              stream);
}

void* GetBasePtr(const std::shared_ptr<Allocation>& allocation) {
  return allocation::AllocatorFacade::Instance().GetBasePtr(allocation);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
uint64_t Release(const platform::CUDAPlace& place, gpuStream_t stream) {
  return allocation::AllocatorFacade::Instance().Release(place, stream);
}

void RecordStream(std::shared_ptr<Allocation> allocation, gpuStream_t stream) {
  return allocation::AllocatorFacade::Instance().RecordStream(allocation,
                                                              stream);
}

gpuStream_t GetStream(const std::shared_ptr<Allocation>& allocation) {
  return allocation::AllocatorFacade::Instance().GetStream(allocation);
}

#endif
}  // namespace memory
}  // namespace paddle
