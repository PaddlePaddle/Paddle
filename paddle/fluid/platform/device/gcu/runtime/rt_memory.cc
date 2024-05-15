/* Copyright (c) 2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/gcu/runtime/rt_memory.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/platform/device/gcu/runtime/rt_resources.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

struct MemoryImpl {
  uint8_t *gcu_mem;
  uint64_t nbytes;
  int dev_id;

  static std::shared_ptr<MemoryImpl> CreateMemory(uint64_t nbytes, int dev_id) {
    GcuDeviceGuard guard(dev_id);
    uint8_t *gcu_mem;
    auto ret = topsMalloc(reinterpret_cast<void **>(&gcu_mem), nbytes);
    if (ret != topsSuccess) {
      return nullptr;
    }
    return std::make_shared<MemoryImpl>(gcu_mem, nbytes, dev_id);
  }

  MemoryImpl(uint8_t *gcu_mem, uint64_t nbytes, int dev_id)
      : gcu_mem(gcu_mem), nbytes(nbytes), dev_id(dev_id) {
    std::string device = "GCU" + std::to_string(dev_id);
    const auto alloc_nbytes = static_cast<int64_t>(nbytes);
    ResourceMgr::GetInstance()->RTCounter(device + "_Memory", 1);
    ResourceMgr::GetInstance()->RTCounter(device + "_MemoryUse", alloc_nbytes);
  }

  ~MemoryImpl() {
    std::string device = "GCU" + std::to_string(dev_id);
    RT_CHECK_NO_THROW(topsFree(gcu_mem));
    const auto free_nbytes = static_cast<int64_t>(nbytes);
    ResourceMgr::GetInstance()->RTCounter(device + "_Memory", -1);
    ResourceMgr::GetInstance()->RTCounter(device + "_MemoryUse", -free_nbytes);
  }

  MemoryImpl() = delete;
  RT_DISALLOW_COPY_AND_ASSIGN(MemoryImpl);
};

Memory::Memory(Context *ctx,
               std::shared_ptr<MemoryImpl> impl,
               uint64_t offset,
               uint64_t nbytes)
    : ctx(ctx), mem_impl(impl), offset(offset), nbytes(nbytes), dims_({}) {}

Memory::~Memory() {}

std::shared_ptr<Memory> Memory::CreateMemory(Context *ctx, uint64_t nbytes) {
  nbytes = (nbytes == 0 ? 1 : nbytes);
  std::shared_ptr<MemoryImpl> impl =
      MemoryImpl::CreateMemory(nbytes, ctx->device);
  if (impl == nullptr) {
    return nullptr;
  }
  return std::make_shared<Memory>(ctx, impl, 0, nbytes);
}

std::shared_ptr<Memory> Memory::CreateSubMemory(std::shared_ptr<Memory> parent,
                                                uint64_t offset,
                                                uint64_t size) {
  PADDLE_ENFORCE_GE(parent->nbytes,
                    (offset + size),
                    platform::errors::Unavailable(
                        "Not have enough memory to create submemory, "
                        "parent->nbyte:%lu, offset:%lu, size:%lu",
                        parent->nbytes,
                        offset,
                        size));
  return std::make_shared<Memory>(parent->ctx, parent->mem_impl, offset, size);
}

void Memory::SetDims(const std::vector<int64_t> &dims, bool dynamic) {
  if (dynamic) {
    RT_CHECK(topsMemorySetDims(mem_impl->gcu_mem + offset,
                               const_cast<int64_t *>(dims.data()),
                               dims.size()));
  } else {
    dims_ = dims;
  }
}

std::vector<int64_t> Memory::GetDims(bool dynamic) const {
  if (dynamic) {
    size_t ndims = 16;
    std::vector<int64_t> dims(ndims, 0);
    RT_CHECK(topsMemoryGetDims(mem_impl->gcu_mem + offset,
                               reinterpret_cast<int64_t *>(dims.data()),
                               &ndims));
    PADDLE_ENFORCE_LT(ndims, 16);
    return std::vector<int64_t>(dims.begin(), dims.begin() + ndims);
  } else {
    return dims_;
  }
}

void *Memory::GetDevAddr() const { return mem_impl->gcu_mem + offset; }

uint64_t Memory::Nbytes() const { return nbytes; }

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
