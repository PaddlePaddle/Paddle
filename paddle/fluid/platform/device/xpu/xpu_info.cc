/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/device/xpu/xpu_info.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/lock_guard_ptr.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/xpu/xpu_info.h"

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
                   const platform::XPUPlace& dst_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(dst_place);
  phi::backends::xpu::MemcpySyncH2D(dst, src, count, dst_place, *dev_ctx);
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUPlace& src_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2H(dst, src, count, src_place, *dev_ctx);
}

// if src.device == dst.device and you need sync , after call this function,
// need to call dev_ctx.Wait()
void MemcpySyncD2D(void* dst,
                   const platform::XPUPlace& dst_place,
                   const void* src,
                   const platform::XPUPlace& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
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

/**************************** buddy allocator **************************/
void XPUMemoryUsage(size_t* available, size_t* total) {
  size_t actual_available, actual_total;
  RecordedXPUMemGetInfo(available,
                        total,
                        &actual_available,
                        &actual_total,
                        GetXPUCurrentDeviceId());
}

constexpr static float fraction_reserve_xpu_memory = 0.05f;
size_t XPUAvailableMemToAlloc() {
  size_t total = 0;
  size_t available = 0;
  XPUMemoryUsage(&available, &total);
  size_t reserving =
      static_cast<size_t>(fraction_reserve_xpu_memory * available);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = available - reserving;
  size_t min_chunk_size = XPUMinChunkSize();
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  VLOG(10) << "XPU usage " << ((total - available) >> 20) << "M/"
           << (total >> 20) << "M, " << (available_to_alloc >> 20)
           << "M available to allocate";
  return available_to_alloc;
}

size_t XPUMaxAllocSize() {
  const char* maxsize = std::getenv("XPUMaxAllocSize");
  size_t max_all_size =
      static_cast<size_t>((maxsize && atoi(maxsize) > 0) ? atoi(maxsize) : 1);
  for (uint32_t i = 0; i < 3; i++) {
    max_all_size *= 1024;
  }
  // XPU max malloc size is 1GB once.
  return std::min(std::max(XPUInitAllocSize(), XPUReallocSize()), max_all_size);
}

static size_t XPUAllocSize(bool realloc) {
  size_t available_to_alloc = XPUAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(
      available_to_alloc,
      0,
      phi::errors::ResourceExhausted("Not enough available XPU memory."));
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? 0.92f : 0ul;
  size_t alloc_bytes =
      (flag_mb > 0ul ? flag_mb << 20 : available_to_alloc * 0.92f);
  PADDLE_ENFORCE_GE(
      available_to_alloc,
      alloc_bytes,
      phi::errors::ResourceExhausted("Not enough available XPU memory."));
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  // return alloc_bytes;
  //  Max Alloc is 1G
  size_t max_all_size = 1;
  for (uint32_t i = 0; i < 3; i++) {
    max_all_size *= 1024;
  }
  return std::min(alloc_bytes, max_all_size);
}

size_t XPUInitAllocSize() { return XPUAllocSize(/* realloc = */ false); }

size_t XPUReallocSize() { return XPUAllocSize(/* realloc = */ true); }

size_t XPUMinChunkSize() { return 1 << 6; }

size_t XPUMaxChunkSize() {
  size_t max_chunk_size = XPUMaxAllocSize();
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;
}

static void RaiseNonOutOfMemoryError(int status) {
  if (status == -705) {
    status = 0;
  }
  PADDLE_ENFORCE_XRE_SUCCESS(status);
}

class RecordedXPUMallocHelper {
 private:
  explicit RecordedXPUMallocHelper(int dev_id, uint64_t limit_size = 0)
      : dev_id_(dev_id), limit_size_(limit_size) {
    if (NeedRecord()) {
      mtx_.reset(new std::mutex());
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
        phi::errors::OutOfRange(
            "Device id must be not less than 0, but got %d.", dev_id));
    PADDLE_ENFORCE_LT(
        dev_id,
        instances_.size(),
        phi::errors::OutOfRange("Device id %d exceeds XPU card number %d.",
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
    xpu_free(ptr);
    cur_size_.fetch_sub(size);
  }

  bool GetMemInfo(size_t* avail,
                  size_t* total,
                  size_t* actual_avail,
                  size_t* actual_total) {
    {
      XPUDeviceGuard guard(dev_id_);
      FILE* fp = NULL;
      char buf[10000] = {0};
      std::string cmd = std::string("xpu_smi -d ") + std::to_string(dev_id_) +
                        std::string(" -m");
      fp = popen(cmd.c_str(), "r");
      if (fp) {
        int ret = fread(buf, 1, sizeof(buf) - 1, fp);
        if (ret > 0) {
          std::string space_delimiter = " ";
          std::vector<std::string> words{};
          std::string buf_string = std::string(buf);
          size_t pos = 0;
          while ((pos = buf_string.find(space_delimiter)) !=
                 std::string::npos) {
            words.push_back(buf_string.substr(0, pos));
            buf_string.erase(0, pos + space_delimiter.length());
          }
          *actual_avail =
              (std::atoi(words[18].c_str()) - std::atoi(words[17].c_str())) *
              1024;
          *actual_total = std::atoi(words[18].c_str()) * 1024;
        }
        pclose(fp);
      }
    }
    *actual_avail *= 1024;
    *actual_total *= 1024;
    VLOG(10) << "actual_avail: " << *actual_avail;
    VLOG(10) << "actual_total: " << *actual_total;
    if (NeedRecord()) {
      std::lock_guard<std::mutex> guard(*mtx_);
      *avail = std::min(*actual_avail, limit_size_ - cur_size_.load());
      *total = std::min(*actual_total, limit_size_);
      return *total < *actual_total;
    } else {
      *avail = *actual_avail;
      *total = *actual_total;
      return false;
    }
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

bool RecordedXPUMemGetInfo(size_t* avail,
                           size_t* total,
                           size_t* actual_avail,
                           size_t* actual_total,
                           int dev_id) {
  return RecordedXPUMallocHelper::Instance(dev_id)->GetMemInfo(
      avail, total, actual_avail, actual_total);
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
