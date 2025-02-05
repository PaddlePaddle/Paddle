// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/platform/device/xpu/xpu_info.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#include "paddle/common/flags.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/memory.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/lock_guard_ptr.h"

namespace paddle {
namespace platform {

/**************************** Version Management **************************/

//! Get the version of XPU Driver
int GetDriverVersion() { return phi::backends::xpu::GetDriverVersion(); }

//! Get the version of XPU Runtime
int GetRuntimeVersion() { return phi::backends::xpu::GetRuntimeVersion(); }

/**************************** Device Management **************************/

int GetXPUDeviceCount() { return phi::backends::xpu::GetXPUDeviceCount(); }

int GetXPUCurrentDeviceId() {
  return phi::backends::xpu::GetXPUCurrentDeviceId();
}

void SetXPUDeviceId(int id) { phi::backends::xpu::SetXPUDeviceId(id); }

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices() {
  // use user specified XPUs in single-node multi-process mode.
  return phi::backends::xpu::GetXPUSelectedDevices();
}

/**************************** Memory Management **************************/

void MemcpySyncH2D(void* dst,
                   const void* src,
                   size_t count,
                   const phi::XPUPlace& dst_place) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(dst_place);
  phi::backends::xpu::MemcpySyncH2D(dst, src, count, dst_place, *dev_ctx);
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const phi::XPUPlace& src_place) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2H(dst, src, count, src_place, *dev_ctx);
}

// if src.device == dst.device and you need sync , after call this function,
// need to call dev_ctx.Wait()
void MemcpySyncD2D(void* dst,
                   const phi::XPUPlace& dst_place,
                   const void* src,
                   const phi::XPUPlace& src_place,
                   size_t count) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}

void XPUStreamSync(xpuStream stream) {
  PADDLE_ENFORCE_XDNN_SUCCESS(xpu_wait(stream), "xpu_wait");
}

/**************************** Others **************************/

phi::backends::xpu::XPUVersion get_xpu_version(int dev_id) {
  return phi::backends::xpu::get_xpu_version(dev_id);
}

void set_xpu_debug_level(int level) {
  phi::backends::xpu::set_xpu_debug_level(level);
}

/**************************** XPU Allocator **************************/
size_t XPUMinChunkSize() { return 1 << 6; }

static void RaiseNonOutOfMemoryError(int status) {
  if (-1 * status == XPUERR_NOMEM) {
    status = XPU_SUCCESS;
  }
  PADDLE_ENFORCE_XRE_SUCCESS(status);
}

void EmptyCache() {
  std::vector<int> devices = GetXPUSelectedDevices();
  for (auto device : devices) {
    memory::Release(phi::XPUPlace(device));
  }
}

class RecordedXPUMallocHelper {
 private:
  explicit RecordedXPUMallocHelper(int dev_id, uint64_t limit_size = 0)
      : dev_id_(dev_id), limit_size_(limit_size) {
    if (NeedRecord()) {
      mtx_ = std::make_unique<std::mutex>();
    }
  }

  DISABLE_COPY_AND_ASSIGN(RecordedXPUMallocHelper);

 public:
  static RecordedXPUMallocHelper* Instance(int dev_id) {
    std::call_once(once_flag_, [] {
      int dev_cnt = GetXPUDeviceCount();
      instances_.reserve(dev_cnt);
      for (int i = 0; i < dev_cnt; ++i) {
        // NOTE(zhiqiu): share the flags with gpu, avoid more flags.
        instances_.emplace_back(new RecordedXPUMallocHelper(i, 0UL << 20));
      }
    });

    PADDLE_ENFORCE_GE(
        dev_id,
        0,
        common::errors::OutOfRange(
            "Device id must be not less than 0, but got %d.", dev_id));
    PADDLE_ENFORCE_LT(
        dev_id,
        instances_.size(),
        common::errors::OutOfRange("Device id %d exceeds XPU card number %d.",
                                   dev_id,
                                   instances_.size()));
    return instances_[dev_id].get();
  }

  /**
   * Try to allocate `size` XPU memory. Only XPUERR_NOMEM
   * or XPU_SUCCESS would be returned.
   */
  int Malloc(void** ptr, size_t size) {
    LockGuardPtr<std::mutex> lock(mtx_);
    if (UNLIKELY(NeedRecord() && cur_size_.load() + size > limit_size_)) {
      return XPUERR_NOMEM;
    }

    XPUDeviceGuard guard(dev_id_);
    VLOG(10) << "Allocate " << size << " bytes with ptr = " << &(ptr);
    auto result = xpu_malloc(ptr, size);
    if (result == XPU_SUCCESS) {
      cur_size_.fetch_add(size);
      DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, size);
      return result;
    } else {
      RaiseNonOutOfMemoryError(result);
      // Non out of memory error would be raised inside
      // RaiseNonOutOfMemoryError. Therefore, we can
      // return XPUERR_NOMEM directly here.
      return XPUERR_NOMEM;
    }
  }

  /**
   * Free XPU memory. Usually, free is not allowed to raise error.
   * If it does raise error, the process should be crashed.
   */
  void Free(void* ptr, size_t size) {
    XPUDeviceGuard guard(dev_id_);
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* dev_ctx = pool.GetByPlace(phi::XPUPlace(dev_id_));
    dev_ctx->Wait();
    xpu_free(ptr);
    cur_size_.fetch_sub(size);
    DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, -size);
  }

  inline bool NeedRecord() const { return limit_size_ != 0; }

  uint64_t RecordedSize() const { return cur_size_.load(); }

  uint64_t LimitSize() const { return limit_size_; }

 private:
  const int dev_id_;
  const uint64_t limit_size_;
  std::atomic<uint64_t> cur_size_{0};

  mutable std::unique_ptr<std::mutex> mtx_;

  static std::once_flag once_flag_;
  static std::vector<std::unique_ptr<RecordedXPUMallocHelper>> instances_;
};

std::once_flag RecordedXPUMallocHelper::once_flag_;
std::vector<std::unique_ptr<RecordedXPUMallocHelper>>
    RecordedXPUMallocHelper::instances_;

int RecordedXPUMalloc(void** ptr, size_t size, int dev_id) {
  return RecordedXPUMallocHelper::Instance(dev_id)->Malloc(ptr, size);
}

void RecordedXPUFree(void* p, size_t size, int dev_id) {
  return RecordedXPUMallocHelper::Instance(dev_id)->Free(p, size);
}

uint64_t RecordedXPUMallocSize(int dev_id) {
  return RecordedXPUMallocHelper::Instance(dev_id)->RecordedSize();
}

uint64_t RecordedXPULimitSize(int dev_id) {
  return RecordedXPUMallocHelper::Instance(dev_id)->LimitSize();
}

bool IsXPUMallocRecorded(int dev_id) {
  return RecordedXPUMallocHelper::Instance(dev_id)->NeedRecord();
}

}  // namespace platform
}  // namespace paddle
