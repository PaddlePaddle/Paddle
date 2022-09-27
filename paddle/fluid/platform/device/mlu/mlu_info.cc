/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/mlu/mlu_info.h"

#include <mutex>
#include <vector>

#include "gflags/gflags.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device/mlu/enforce.h"
#include "paddle/fluid/platform/lock_guard_ptr.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/string/split.h"

DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);
DECLARE_uint64(gpu_memory_limit_mb);

constexpr static float fraction_reserve_mlu_memory = 0.05f;

PADDLE_DEFINE_EXPORTED_string(
    selected_mlus,
    "",
    "A list of device ids separated by comma, like: 0,1,2,3. "
    "This option is useful when doing multi process training and "
    "each process have only one device (MLU). If you want to use "
    "all visible devices, set this to empty string. NOTE: the "
    "reason of doing this is that we want to use P2P communication"
    "between MLU devices, use MLU_VISIBLE_DEVICES can only use"
    "share-memory only.");

USE_MLU_MEM_STAT;
namespace paddle {
namespace platform {

static int GetMLUDeviceCountImpl() {
  int x, y, z;
  // When cnrtDriverGetVersion is executed, the device is initialized,
  // no longer needs to call cnrtInit().
  cnrtStatus stat = cnrtDriverGetVersion(&x, &y, &z);
  if (stat != cnrtSuccess) {
    VLOG(2) << "MLU Driver Version can't be detected. No MLU driver!";
    return 0;
  }

  const auto *mlu_visible_devices = std::getenv("MLU_VISIBLE_DEVICES");
  if (mlu_visible_devices != nullptr) {
    std::string mlu_visible_devices_str(mlu_visible_devices);
    if (std::all_of(mlu_visible_devices_str.begin(),
                    mlu_visible_devices_str.end(),
                    [](char ch) { return ch == ' '; })) {
      VLOG(2) << "MLU_VISIBLE_DEVICES  is set to be "
                 "empty. No MLU detected.";
      return 0;
    }
  }

  int count;
  PADDLE_ENFORCE_MLU_SUCCESS(cnDeviceGetCount(&count));
  return count;
}

int GetMLUDeviceCount() {
  static auto dev_cnt = GetMLUDeviceCountImpl();
  return dev_cnt;
}

std::vector<int> GetMLUSelectedDevices() {
  // use user specified MLUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_mlus.empty()) {
    auto devices_str = paddle::string::Split(FLAGS_selected_mlus, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = GetMLUDeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

void CheckDeviceId(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetMLUDeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than MLU count, "
                        "but received id is: %d. MLU count is: %d.",
                        id,
                        GetMLUDeviceCount()));
}

int GetMLUDriverVersion(int id) {
  CheckDeviceId(id);
  int x, y, z;
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtDriverGetVersion(&x, &y, &z));
  return x * 10000 + y * 100 + z;
}

int GetMLURuntimeVersion(int id) {
  CheckDeviceId(id);
  int x, y, z;
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtGetLibVersion(&x, &y, &z));
  return x * 10000 + y * 100 + z;
}

int GetMLUCnnlVersion(int id) {
  CheckDeviceId(id);
  int x, y, z;
  cnnlGetLibVersion(&x, &y, &z);
  return x * 10000 + y * 100 + z;
}

int GetMLUOpVersion(int id) {
  CheckDeviceId(id);
  int x, y, z;
  mluOpGetLibVersion(&x, &y, &z);
  return x * 10000 + y * 100 + z;
}

int GetMLUCurrentDeviceId() {
  int device_id;
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtGetDevice(&device_id));
  return device_id;
}

void SetMLUDeviceId(int id) {
  CheckDeviceId(id);
  PADDLE_RETRY_MLU_SUCCESS(cnrtSetDevice(id));
}

void GetMLUDeviceHandle(int device_ordinal, mluDeviceHandle *device) {
  cnStatus res = cnDeviceGet(device, device_ordinal);
  if (res != CN_SUCCESS) {
    VLOG(2) << "failed to get handle of MLU Device.";
  }
  PADDLE_ENFORCE_MLU_SUCCESS(res);
}

int GetMLUComputeCapability(int id) {
  CheckDeviceId(id);
  mluDeviceHandle device;
  GetMLUDeviceHandle(id, &device);

  int major, minor;
  cnStatus major_stat = cnDeviceGetAttribute(
      &major, CN_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cnStatus minor_stat = cnDeviceGetAttribute(
      &minor, CN_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  PADDLE_ENFORCE_MLU_SUCCESS(major_stat);
  PADDLE_ENFORCE_MLU_SUCCESS(minor_stat);

  return major * 10 + minor;
}

void MLUMemoryUsage(size_t *available, size_t *total) {
  size_t actual_available, actual_total;
  RecordedMLUMemGetInfo(available,
                        total,
                        &actual_available,
                        &actual_total,
                        platform::GetMLUCurrentDeviceId());
}

size_t MLUAvailableMemToAlloc() {
  size_t total = 0;
  size_t available = 0;
  MLUMemoryUsage(&available, &total);
  size_t reserving =
      static_cast<size_t>(fraction_reserve_mlu_memory * available);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = available - reserving;
  size_t min_chunk_size = MLUMinChunkSize();
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  VLOG(10) << "MLU usage " << ((total - available) >> 20) << "M/"
           << (total >> 20) << "M, " << (available_to_alloc >> 20)
           << "M available to allocate";
  return available_to_alloc;
}

size_t MLUMaxAllocSize() {
  return std::max(MLUInitAllocSize(), MLUReallocSize());
}

static size_t MLUAllocSize(bool realloc) {
  size_t available_to_alloc = MLUAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(
      available_to_alloc,
      0,
      platform::errors::ResourceExhausted("Not enough available MLU memory."));
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_gpu_memory_in_mb
                           : FLAGS_initial_gpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul
           ? flag_mb << 20
           : available_to_alloc * FLAGS_fraction_of_gpu_memory_to_use);
  PADDLE_ENFORCE_GE(
      available_to_alloc,
      alloc_bytes,
      platform::errors::ResourceExhausted("Not enough available MLU memory."));
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  return alloc_bytes;
}

size_t MLUInitAllocSize() { return MLUAllocSize(/* realloc = */ false); }

size_t MLUReallocSize() { return MLUAllocSize(/* realloc = */ true); }

size_t MLUMinChunkSize() {
  // Allow to allocate the minimum chunk size is 256 bytes.
  return 1 << 8;
}

size_t MLUMaxChunkSize() {
  size_t max_chunk_size = MLUMaxAllocSize();
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;
}

void MLUMemcpyD2HAsync(void *dst,
                       const void *src,
                       size_t num,
                       mluStream stream) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemcpyAsync(
      dst, const_cast<void *>(src), num, stream, cnrtMemcpyDevToHost));
}

void MLUMemcpyD2HSync(void *dst, const void *src, size_t num) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnrtMemcpy(dst, const_cast<void *>(src), num, cnrtMemcpyDevToHost));
}

void MLUMemcpyH2DAsync(void *dst,
                       const void *src,
                       size_t num,
                       mluStream stream) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemcpyAsync(
      dst, const_cast<void *>(src), num, stream, cnrtMemcpyHostToDev));
}
void MLUMemcpyH2DSync(void *dst, const void *src, size_t num) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnrtMemcpy(dst, const_cast<void *>(src), num, cnrtMemcpyHostToDev));
}

void MLUMemcpyD2DAsync(void *dst,
                       const void *src,
                       size_t num,
                       mluStream stream) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemcpyAsync(
      dst, const_cast<void *>(src), num, stream, cnrtMemcpyDevToDev));
}
void MLUMemcpyD2DSync(void *dst, const void *src, size_t num) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnrtMemcpy(dst, const_cast<void *>(src), num, cnrtMemcpyDevToDev));
}

void MLUMemcpyPeerAsync(void *dst,
                        int dst_device,
                        const void *src,
                        int src_device,
                        size_t num,
                        mluStream stream) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemcpyPeerAsync(
      dst, dst_device, const_cast<void *>(src), src_device, num, stream));
}

void MLUMemcpyPeerSync(
    void *dst, int dst_device, const void *src, int src_device, size_t num) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemcpyPeer(
      dst, dst_device, const_cast<void *>(src), src_device, num));
}

void MLUMemsetAsync(void *dst, int value, size_t count, mluStream stream) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemsetAsync(dst, value, count, stream));
}

void MLUStreamSync(mluStream stream) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueSync(stream));
}

static void RaiseNonOutOfMemoryError(cnrtStatus *status) {
  if (*status == cnrtErrorNoMem) {
    *status = cnrtSuccess;
  }
  PADDLE_ENFORCE_MLU_SUCCESS(*status);

  *status = cnrtGetLastError();
  if (*status == cnrtErrorNoMem) {
    *status = cnrtSuccess;
  }
  PADDLE_ENFORCE_MLU_SUCCESS(*status);
}

class RecordedMLUMallocHelper {
 private:
  explicit RecordedMLUMallocHelper(int dev_id, uint64_t limit_size = 0)
      : dev_id_(dev_id), limit_size_(limit_size) {
    if (NeedRecord()) {
      mtx_.reset(new std::mutex());
    }
  }

  DISABLE_COPY_AND_ASSIGN(RecordedMLUMallocHelper);

 public:
  static RecordedMLUMallocHelper *Instance(int dev_id) {
    std::call_once(once_flag_, [] {
      int dev_cnt = GetMLUDeviceCount();
      instances_.reserve(dev_cnt);
      for (int i = 0; i < dev_cnt; ++i) {
        instances_.emplace_back(
            new RecordedMLUMallocHelper(i, FLAGS_gpu_memory_limit_mb << 20));
      }
    });

    PADDLE_ENFORCE_GE(
        dev_id,
        0,
        platform::errors::OutOfRange(
            "Device id must be not less than 0, but got %d.", dev_id));
    PADDLE_ENFORCE_LT(
        dev_id,
        instances_.size(),
        platform::errors::OutOfRange("Device id %d exceeds mlu card number %d.",
                                     dev_id,
                                     instances_.size()));
    return instances_[dev_id].get();
  }

  /**
   * Try to allocate `size` mlu memory. Only cnrtErrorNoMem
   * or cnrtSuccess would be returned, and the cnrtGetLastError() flag
   * would be clear.
   */
  cnrtStatus Malloc(void **ptr, size_t size) {
    LockGuardPtr<std::mutex> lock(mtx_);
    if (UNLIKELY(NeedRecord() && cur_size_.load() + size > limit_size_)) {
      return cnrtErrorNoMem;
    }

    MLUDeviceGuard guard(dev_id_);
    auto result = cnrtMalloc(ptr, size);
    if (result == cnrtSuccess) {
      cur_size_.fetch_add(size);
      STAT_INT_ADD("STAT_mlu" + std::to_string(dev_id_) + "_mem_size", size);
      return cnrtSuccess;
    } else {
      RaiseNonOutOfMemoryError(&result);
      // Non out of memory error would be raised inside
      // RaiseNonOutOfMemoryError.
      // Therefore, we can return cnrtErrorNoMem directly here.
      return cnrtErrorNoMem;
    }
  }

  /**
   * Free mlu memory. Usually, free is not allowed to raise error.
   * If it does raise error, the process should be crashed.
   */
  void Free(void *ptr, size_t size) {
    MLUDeviceGuard guard(dev_id_);
    auto err = cnrtFree(ptr);
    PADDLE_ENFORCE_MLU_SUCCESS(err);
    if (NeedRecord()) {
      cur_size_.fetch_sub(size);
    }
    STAT_INT_SUB("STAT_mlu" + std::to_string(dev_id_) + "_mem_size", size);
  }

  bool GetMemInfo(size_t *avail,
                  size_t *total,
                  size_t *actual_avail,
                  size_t *actual_total) {
    {
      MLUDeviceGuard guard(dev_id_);
      auto result = cnrtMemGetInfo(actual_avail, actual_total);
      if (result != cnrtSuccess) {
        *actual_avail = 0;
      }
      RaiseNonOutOfMemoryError(&result);
    }

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
  static std::vector<std::unique_ptr<RecordedMLUMallocHelper>> instances_;
};  // NOLINT

std::once_flag RecordedMLUMallocHelper::once_flag_;
std::vector<std::unique_ptr<RecordedMLUMallocHelper>>
    RecordedMLUMallocHelper::instances_;

cnrtStatus RecordedMLUMalloc(void **ptr, size_t size, int dev_id) {
  return RecordedMLUMallocHelper::Instance(dev_id)->Malloc(ptr, size);
}

void RecordedMLUFree(void *p, size_t size, int dev_id) {
  return RecordedMLUMallocHelper::Instance(dev_id)->Free(p, size);
}

bool RecordedMLUMemGetInfo(size_t *avail,
                           size_t *total,
                           size_t *actual_avail,
                           size_t *actual_total,
                           int dev_id) {
  return RecordedMLUMallocHelper::Instance(dev_id)->GetMemInfo(
      avail, total, actual_avail, actual_total);
}

uint64_t RecordedMLUMallocSize(int dev_id) {
  return RecordedMLUMallocHelper::Instance(dev_id)->RecordedSize();
}

bool IsMLUMallocRecorded(int dev_id) {
  return RecordedMLUMallocHelper::Instance(dev_id)->NeedRecord();
}

void EmptyCache(void) {
  std::vector<int> devices = GetMLUSelectedDevices();
  for (auto device : devices) {
    memory::Release(MLUPlace(device));
  }
}

}  // namespace platform
}  // namespace paddle
