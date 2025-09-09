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

#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif

#include <mutex>  // NOLINT

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"

#if CUDA_VERSION >= 10020

namespace paddle {
namespace memory {
namespace allocation {

// Allocate memory using NVIDIA's virtual memory management technology

struct VmmShareInfo {
  int os_fd{-1};     // Linux: file descriptor
  size_t size{0};    // total mapped length (page aligned)
  size_t offset{0};  // byte offset inside the handle (page aligned)
  int device{-1};    // exporter device
};

class CUDAVirtualMemAllocator : public Allocator {
 public:
  explicit CUDAVirtualMemAllocator(const phi::GPUPlace& place);
  ~CUDAVirtualMemAllocator() override;
  bool IsAllocThreadSafe() const override;
  static bool TryExportShareHandle(int device,
                                   void* base_ptr,
                                   VmmShareInfo* out);
  // —— 新增：静态注册/查找（每设备可能多个实例）
  static std::shared_ptr<Allocation> ImportShareHandle(const VmmShareInfo& in);
  static void Register(int device, CUDAVirtualMemAllocator* a);
  static void Unregister(int device, CUDAVirtualMemAllocator* a);

 protected:
  void FreeImpl(phi::Allocation* allocation) override;
  phi::Allocation* AllocateImpl(size_t size) override;

  // —— 新增：从任意 VA 做“区间命中”并导出句柄
  bool ExportShareHandleFromVA(CUdeviceptr va, VmmShareInfo* out);

 private:
  phi::GPUPlace place_;
  std::once_flag once_flag_;

  CUdeviceptr virtual_mem_base_;
  size_t virtual_mem_size_;
  size_t virtual_mem_alloced_offset_;
  size_t granularity_;

  CUmemAllocationProp prop_;
  std::vector<CUmemAccessDesc> access_desc_;

  std::map<CUdeviceptr, std::pair<CUmemGenericAllocationHandle, size_t>>
      virtual_2_physical_map_;

  // registry (device -> allocator) so pybind can find us to export
  inline static std::mutex s_reg_mu_;
  // —— 新增：每设备的分配器注册表
  inline static std::map<int, std::vector<CUDAVirtualMemAllocator*>> s_regs_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
