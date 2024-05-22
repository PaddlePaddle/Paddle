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

#pragma once

#include <memory>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/stream.h"

namespace paddle {
namespace memory {

using allocation::AllocationPtr;
using allocation::Allocator;
using phi::Allocation;

extern std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                               size_t size);

extern AllocationPtr Alloc(const platform::Place& place, size_t size);

extern uint64_t Release(const platform::Place& place);

extern std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                               size_t size,
                                               const phi::Stream& stream);

extern AllocationPtr Alloc(const platform::CUDAPlace& place,
                           size_t size,
                           const phi::Stream& stream);

extern bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                         const phi::Stream& stream);

extern void* GetBasePtr(const std::shared_ptr<Allocation>& allocation);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
extern uint64_t Release(const platform::CUDAPlace& place, gpuStream_t stream);

void RecordStream(std::shared_ptr<Allocation> allocation, gpuStream_t stream);

void EraseStream(std::shared_ptr<Allocation> allocation, gpuStream_t stream);

gpuStream_t GetStream(const std::shared_ptr<Allocation>& allocation);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
void RecordStream(std::shared_ptr<Allocation> allocation,
                  phi::stream::stream_t stream);
#endif

template <typename StreamType>
struct ThrustAllocator {
  typedef char value_type;
  ThrustAllocator(platform::Place place, StreamType stream) {
    VLOG(2) << "construct allocator";
    place_ = place;
    stream_ = stream;
  }
  ~ThrustAllocator() { VLOG(2) << "destory allocator"; }
  char* allocate(std::ptrdiff_t num_bytes) {
    VLOG(2) << "allocate " << num_bytes << " bytes";
    auto storage = memory::AllocShared(
        place_,
        num_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(stream_)));
    char* ptr = reinterpret_cast<char*>(storage->ptr());
    busy_allocation_.emplace(std::make_pair(ptr, storage));
    return ptr;
  }
  void deallocate(char* ptr, size_t) {
    VLOG(2) << "deallocate ";
    allocation_map_type::iterator iter = busy_allocation_.find(ptr);
    CHECK(iter != busy_allocation_.end());
    busy_allocation_.erase(iter);
  }

 private:
  typedef std::unordered_map<char*, std::shared_ptr<phi::Allocation>>
      allocation_map_type;
  allocation_map_type busy_allocation_;
  platform::Place place_;
  StreamType stream_;
};

}  // namespace memory
}  // namespace paddle
