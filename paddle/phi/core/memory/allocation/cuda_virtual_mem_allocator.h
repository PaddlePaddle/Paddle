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
#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif

#include <memory>
#include <mutex>  // NOLINT
#include <vector>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"

#ifdef PADDLE_WITH_CUDA

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

struct ImportedVmmMulti {
  CUdeviceptr base{0};
  size_t reserved_size{0};
  std::vector<CUmemGenericAllocationHandle> hs;
  ~ImportedVmmMulti() {
    if (base && reserved_size) {
      phi::dynload::cuMemUnmap(base, reserved_size);
    }
    for (auto h : hs) {
      if (h) {
        phi::dynload::cuMemRelease(h);
      }
    }
    if (base && reserved_size) {
      phi::dynload::cuMemAddressFree(base, reserved_size);
    }
  }
};

class VmmImportedAllocation : public phi::Allocation {
 public:
  VmmImportedAllocation(void* ptr,
                        size_t bytes,
                        phi::Place place,
                        std::shared_ptr<ImportedVmmMulti> keep)
      : Allocation(ptr, bytes, place), keep_(std::move(keep)) {}

 private:
  std::shared_ptr<ImportedVmmMulti> keep_;
};

#pragma pack(push, 1)
struct VmmIpcHeader {
  uint8_t version;
  uint8_t type;
  uint16_t flags;
  uint32_t pid;
  uint32_t num_entries;
  uint64_t alloc_size;
  uint64_t offset;
  uint64_t reserved_size;
};

struct VmmIpcEntry {
  uint8_t handle_type;
  uint8_t reserved[7];
  uint64_t rel_offset;
  uint64_t chunk_size;
  uint64_t chunk_rel_off;
};
#pragma pack(pop)

static_assert(sizeof(VmmIpcHeader) == 36, "VmmIpcHeader size changed");
static_assert(sizeof(VmmIpcEntry) == 32, "VmmIpcEntry size changed");

class CUDAVirtualMemAllocator : public Allocator {
 public:
  explicit CUDAVirtualMemAllocator(const phi::GPUPlace& place);

  bool IsAllocThreadSafe() const override;

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

  void InitOnce();
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
