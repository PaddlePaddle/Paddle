// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <memory>
#include <vector>

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cuda_driver.h"
using VmmDevicePtr = CUdeviceptr;
using VmmAllocHandle = CUmemGenericAllocationHandle;
#else
using VmmDevicePtr = uintptr_t;
using VmmAllocHandle = uint64_t;
#endif

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

struct ImportedVmmMulti {
  VmmDevicePtr base{0};
  size_t reserved_size{0};
  std::vector<VmmAllocHandle> hs;
#if defined(PADDLE_WITH_CUDA)
  ~ImportedVmmMulti() {
    if (base && reserved_size) {
      phi::dynload::cuMemUnmap(base, reserved_size);
    }
    for (auto h : hs) {
      if (h) phi::dynload::cuMemRelease(h);
    }
    if (base && reserved_size) {
      phi::dynload::cuMemAddressFree(base, reserved_size);
    }
  }
#else
  ~ImportedVmmMulti() = default;
#endif
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

struct VmmChunkMeta {
  VmmDevicePtr base;
  size_t size;
  VmmAllocHandle handle;
  int device;
};

struct BlockPart {
  std::shared_ptr<VmmChunkMeta> chunk;
  size_t chunk_rel_off;
  size_t len;
};

#pragma pack(push, 1)
struct VmmIpcHeader {
  uint8_t version;
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

static_assert(sizeof(VmmIpcHeader) == 35, "VmmIpcHeader size changed");
static_assert(sizeof(VmmIpcEntry) == 32, "VmmIpcEntry size changed");

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
