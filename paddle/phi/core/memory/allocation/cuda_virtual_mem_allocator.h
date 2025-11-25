// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>

#include "paddle/phi/backends/dynload/cuda_driver.h"
#include "paddle/phi/core/memory/allocation/vmm_ipc_allocation.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif

#include <mutex>  // NOLINT
#include <unordered_map>
#include <vector>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"

#ifdef PADDLE_WITH_CUDA

namespace paddle {
namespace memory {
namespace allocation {

// Allocate memory using NVIDIA's virtual memory management technology
class CUDAVirtualMemAllocator : public Allocator {
 public:
  explicit CUDAVirtualMemAllocator(const phi::GPUPlace& place);

  bool IsAllocThreadSafe() const override;

  static CUmemGenericAllocationHandle GetHandleFromBasePtr(void* base_ptr);

 protected:
  void FreeImpl(phi::Allocation* allocation) override;
  phi::Allocation* AllocateImpl(size_t size) override;

 private:
  phi::GPUPlace place_;
  std::once_flag init_flag_;

  CUdeviceptr virtual_mem_base_;
  size_t virtual_mem_size_;
  size_t virtual_mem_alloced_offset_;
  size_t granularity_;

  CUmemAllocationProp prop_;
  std::vector<CUmemAccessDesc> access_desc_;

  std::map<CUdeviceptr, std::pair<CUmemGenericAllocationHandle, size_t>>
      virtual_2_physical_map_;
  static std::mutex base_ptr_handle_mu_;
  static std::unordered_map<void*, CUmemGenericAllocationHandle>
      base_ptr_handle_map_;

  void InitOnce();
  static void RegisterHandle(void* base_ptr, CUmemGenericAllocationHandle h);
  static void UnregisterHandle(void* base_ptr);
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
