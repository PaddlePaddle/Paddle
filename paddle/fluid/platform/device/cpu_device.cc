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

#include "paddle/fluid/platform/device/device_manager.h"
#include "paddle/fluid/platform/device/event.h"
#include "paddle/fluid/platform/device/stream.h"

namespace paddle {
namespace platform {

class CpuDevice : public DeviceInterface {
 public:
  CpuDevice(const std::string& type, int priority, bool is_pluggable)
      : DeviceInterface(type, priority, is_pluggable) {}
  ~CpuDevice() {}

  size_t VisibleDevicesCount() override { return 1; }

  void SetDevice(size_t dev_id) override {}

  void MemoryCopy(size_t dev_id /* note: dst_place */, void* dst,
                  const void* src, size_t size, MemoryCpyKind kind,
                  const stream::Stream* stream = nullptr) override {
    std::memcpy(dst, src, size);
  }

  void* MemoryAllocate(
      size_t dev_id, size_t size,
      MemoryAllocKind kind = MemoryAllocKind::Normal) override {
    return malloc(size);
  }

  void MemoryDeallocate(
      size_t dev_id, void* ptr, size_t size,
      MemoryAllocKind kind = MemoryAllocKind::Normal) override {
    free(ptr);
  }

  void MemorySet(size_t dev_id, void* ptr, uint8_t value,
                 size_t size) override {
    memset(ptr, value, size);
  }

  void MemoryStats(size_t dev_id, size_t* total, size_t* free) override {
    PADDLE_THROW(platform::errors::Unavailable(
        "MemoryStats is not implemented on cpu device."));
  }

  size_t GetMinChunkSize(size_t dev_id) override {
    // Allow to allocate the minimum chunk size is 4 KB.
    constexpr size_t min_chunk_size = 1 << 12;
    VLOG(10) << Type() + " min chunk size " << min_chunk_size;
    return min_chunk_size;
  }
};

}  // namespace platform
}  // namespace paddle

REGISTER_BUILTIN_DEVICE(cpu, paddle::platform::CpuDevice);
