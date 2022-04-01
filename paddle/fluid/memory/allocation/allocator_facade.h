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

#pragma once
#include <memory>
#include "paddle/fluid/memory/allocation/allocator.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/memory/allocation/npu_pinned_allocator.h"
#endif
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#endif
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/stream.h"

namespace paddle {
namespace memory {
namespace allocation {

#ifdef PADDLE_WITH_ASCEND_CL
using NPUPinnedAllocator = paddle::memory::allocation::NPUPinnedAllocator;
#endif

// Allocator Facade is the interface exposed to other modules.
// All the configuration or dirty code under development should
// be hidden behind this facade.
//
// NOTE(yy): This class is a singleton class.
// NOTE(yy): To create a stable ABI and make compilation faster. Here we use
// a Pimpl trick;
class AllocatorFacadePrivate;
class AllocatorFacade {
 public:
  using Allocation = phi::Allocation;
  AllocatorFacade(const AllocatorFacade& o) = delete;
  const AllocatorFacade& operator=(const AllocatorFacade& o) = delete;
  ~AllocatorFacade();

  static AllocatorFacade& Instance();

  AllocatorFacadePrivate* GetPrivate() const;

  const std::shared_ptr<Allocator>& GetAllocator(const platform::Place& place);

  void* GetBasePtr(const std::shared_ptr<Allocation>& allocation);

  const std::shared_ptr<Allocator>& GetZeroAllocator(
      const platform::Place& place);

  // Allocate a shared allocation.
  std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                          size_t size);
  // Allocate a unique allocation.
  AllocationPtr Alloc(const platform::Place& place, size_t size);
  // Release unused memory pool.
  uint64_t Release(const platform::Place& place);

  std::shared_ptr<Allocation> AllocShared(const platform::Place& place,
                                          size_t size,
                                          const phi::Stream& stream);

  AllocationPtr Alloc(const platform::Place& place, size_t size,
                      const phi::Stream& stream);

  bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                    const phi::Stream& stream);

  bool IsStreamSafeCUDAAllocatorUsed();

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // TODO(zhiqiu): change gpuStream_t to phi::Stream if needed.
  uint64_t Release(const platform::CUDAPlace& place, const gpuStream_t& stream);
  void RecordStream(std::shared_ptr<Allocation> allocation,
                    const gpuStream_t& stream);
  const std::shared_ptr<Allocator>& GetAllocator(const platform::Place& place,
                                                 const gpuStream_t& stream);
  const gpuStream_t GetStream(
      const std::shared_ptr<Allocation>& allocation) const;
  void SetDefaultStream(const platform::CUDAPlace& place,
                        const gpuStream_t& stream);
#endif

#ifdef PADDLE_WITH_CUDA
  void PrepareMemoryPoolForCUDAGraph(CUDAGraphID id);
  void RemoveMemoryPoolOfCUDAGraph(CUDAGraphID id);
#endif

  // TODO(yy): Allocate a Copy-On-Write allocation?
 private:
  AllocatorFacade();
  AllocatorFacadePrivate* m_;
#ifdef PADDLE_WITH_CUDA
  std::unordered_map<CUDAGraphID, std::unique_ptr<AllocatorFacadePrivate>>
      cuda_graph_map_;
#endif
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
