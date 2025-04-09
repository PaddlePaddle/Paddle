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

#include "paddle/phi/core/memory/malloc.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/stream.h"

namespace paddle::memory {

std::shared_ptr<Allocation> AllocShared(const phi::Place& place, size_t size) {
  return allocation::AllocatorFacade::Instance().AllocShared(place, size);
}

AllocationPtr Alloc(const phi::Place& place, size_t size) {
  return allocation::AllocatorFacade::Instance().Alloc(place, size);
}

uint64_t Release(const phi::Place& place) {
  return allocation::AllocatorFacade::Instance().Release(place);
}

std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                        size_t size,
                                        const phi::Stream& stream) {
  return allocation::AllocatorFacade::Instance().AllocShared(
      place, size, stream);
}

AllocationPtr Alloc(const phi::Place& place,
                    size_t size,
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
uint64_t Release(const phi::GPUPlace& place, gpuStream_t stream) {
  return allocation::AllocatorFacade::Instance().Release(place, stream);
}

bool RecordStream(std::shared_ptr<Allocation> allocation, gpuStream_t stream) {
  return allocation::AllocatorFacade::Instance().RecordStream(allocation,
                                                              stream);
}

void EraseStream(std::shared_ptr<Allocation> allocation, gpuStream_t stream) {
  return allocation::AllocatorFacade::Instance().EraseStream(allocation,
                                                             stream);
}

gpuStream_t GetStream(const std::shared_ptr<Allocation>& allocation) {
  return allocation::AllocatorFacade::Instance().GetStream(allocation);
}

#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
bool RecordStream(std::shared_ptr<Allocation> allocation,
                  phi::stream::stream_t stream) {
  return allocation::AllocatorFacade::Instance().RecordStream(allocation,
                                                              stream);
}
#endif
}  // namespace paddle::memory
