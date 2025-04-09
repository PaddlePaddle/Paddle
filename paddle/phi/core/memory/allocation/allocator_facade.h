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

#include "paddle/phi/core/memory/allocation/allocator.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#endif
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/stream.h"

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/memory/allocation/custom_allocator.h"
#endif

namespace paddle {
namespace memory {
namespace allocation {

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

  TEST_API static AllocatorFacade& Instance();

  AllocatorFacadePrivate* GetPrivate() const;

  TEST_API const std::shared_ptr<Allocator>& GetAllocator(
      const phi::Place& place);

  TEST_API const std::shared_ptr<Allocator>& GetAutoGrowthAllocator(
      const phi::Place& place);

  void* GetBasePtr(const std::shared_ptr<Allocation>& allocation);

  const std::shared_ptr<Allocator>& GetZeroAllocator(const phi::Place& place);

  // Allocate a shared allocation.
  std::shared_ptr<Allocation> AllocShared(const phi::Place& place, size_t size);
  // Allocate a unique allocation.
  AllocationPtr Alloc(const phi::Place& place, size_t size);
  // Release unused memory pool.
  uint64_t Release(const phi::Place& place);

  std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                          size_t size,
                                          const phi::Stream& stream);

  AllocationPtr Alloc(const phi::Place& place,
                      size_t size,
                      const phi::Stream& stream);

  bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                    const phi::Stream& stream);

  bool IsStreamSafeCUDAAllocatorUsed();
  bool IsCUDAMallocAsyncAllocatorUsed();

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // TODO(zhiqiu): change gpuStream_t to phi::Stream if needed.
  uint64_t Release(const phi::GPUPlace& place, gpuStream_t stream);
  bool RecordStream(std::shared_ptr<Allocation> allocation, gpuStream_t stream);
  void EraseStream(std::shared_ptr<Allocation> allocation, gpuStream_t stream);

  TEST_API const std::shared_ptr<Allocator>& GetAllocator(
      const phi::Place& place, gpuStream_t stream);
  gpuStream_t GetStream(const std::shared_ptr<Allocation>& allocation) const;
  void SetDefaultStream(const phi::GPUPlace& place, gpuStream_t stream);
#elif defined(PADDLE_WITH_XPU)
  TEST_API const std::shared_ptr<Allocator>& GetAllocator(
      const phi::Place& place, XPUStream stream);
  void SetDefaultStream(const phi::XPUPlace& place, XPUStream stream);
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void PrepareMemoryPoolForCUDAGraph(int64_t id);
  void RemoveMemoryPoolOfCUDAGraph(int64_t id);
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  uint64_t Release(const phi::CustomPlace& place, phi::stream::stream_t stream);
  bool RecordStream(std::shared_ptr<Allocation> allocation,
                    phi::stream::stream_t stream);
  TEST_API const std::shared_ptr<Allocator>& GetAllocator(
      const phi::Place& place, phi::stream::stream_t stream);
  phi::stream::stream_t GetStream(
      const std::shared_ptr<Allocation>& allocation) const;
  void SetDefaultStream(const phi::CustomPlace& place,
                        phi::stream::stream_t stream);
#endif
  // TODO(yy): Allocate a Copy-On-Write allocation?
 private:
  AllocatorFacade();
  AllocatorFacadePrivate* m_;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::unordered_map<int64_t, std::unique_ptr<AllocatorFacadePrivate>>
      cuda_graph_map_;
  std::unordered_map<int64_t, int64_t> cuda_graph_ref_cnt_;
#endif
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
