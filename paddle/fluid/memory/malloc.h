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
#include "paddle/fluid/platform/stream/stream.h"

namespace paddle {

namespace platform {
class DeviceContext;
}  // platform

namespace memory {

using allocation::Allocation;
using allocation::Allocator;
using allocation::AllocationPtr;

extern std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                               size_t size);

extern AllocationPtr Alloc(const platform::Place& place, size_t size);

extern AllocationPtr Alloc(const platform::DeviceContext& dev_ctx, size_t size);

extern uint64_t Release(const platform::Place& place);

extern std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                               size_t size,
                                               const platform::Stream& stream);

extern bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                         const platform::Stream& stream);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
extern AllocationPtr Alloc(const platform::CUDAPlace& place, size_t size,
                           const gpuStream_t& stream);

extern uint64_t Release(const platform::CUDAPlace& place,
                        const gpuStream_t& stream);

void RecordStream(std::shared_ptr<Allocation> allocation,
                  const gpuStream_t& stream);

const gpuStream_t& GetStream(const std::shared_ptr<Allocation>& allocation);
#endif
}  // namespace memory
}  // namespace paddle
