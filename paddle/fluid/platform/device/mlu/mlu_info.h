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

#pragma once

#ifdef PADDLE_WITH_MLU
#include <cn_api.h>
#include <cndrv_id.h>
#include <cnnl.h>
#include <cnpapi.h>
#include <cnrt.h>
#ifdef PADDLE_WITH_CNCL
#include <cncl.h>
#endif
#include <vector>

namespace paddle {

using cnStatus = CNresult;
using cnrtStatus = cnrtRet_t;
using cnnlStatus = cnnlStatus_t;
#ifdef PADDLE_WITH_CNCL
using cnclStatus = cnclResult_t;
#endif
using mluStream = cnrtQueue_t;
using mluCnnlHandle = cnnlHandle_t;
using mluEventHandle = cnrtNotifier_t;
using mluDeviceHandle = CNdev;

namespace platform {

//! Get the driver version of the ith MLU.
int GetMLUDriverVersion(int id);

//! Get the runtime version of the ith MLU.
int GetMLURuntimeVersion(int id);

//! Get the total number of MLU devices in system.
int GetMLUDeviceCount();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetMLUSelectedDevices();

//! Get the current MLU device id in system.
int GetMLUCurrentDeviceId();

//! Set the MLU device id for next execution.
void SetMLUDeviceId(int device_id);

//! Get a handle of device ids.
void GetMLUDeviceHandle(int device_ordinal, mluDeviceHandle* device);

//! Get the compute capability of the ith MLU (format: major * 10 + minor)
int GetMLUComputeCapability(int id);

//! Get the memory usage of current MLU device.
void MLUMemoryUsage(size_t* available, size_t* total);

//! Get the available memory to allocate, which is the size of available mlu
//! minus reserving.
size_t MLUAvailableMemToAlloc();

//! Get the maximum allocation size of current MLU device.
size_t MLUMaxAllocSize();

//! Get the initial allocation size of current MLU device.
size_t MLUInitAllocSize();

//! Get the re-allocation size of current MLU device.
size_t MLUReallocSize();

//! Get the minimum chunk size for MLU buddy allocator.
size_t MLUMinChunkSize();

//! Get the maximum chunk size for MLU buddy allocator.
size_t MLUMaxChunkSize();

//! Copy memory from address device to host asynchronously.
void MLUMemcpyD2HAsync(void* dst, const void* src, size_t num,
                       mluStream stream);

//! Copy memory from address device to host synchronously.
void MLUMemcpyD2HSync(void* dst, const void* src, size_t num);

//! Copy memory from address host to device asynchronously.
void MLUMemcpyH2DAsync(void* dst, const void* src, size_t num,
                       mluStream stream);

//! Copy memory from address host to device synchronously.
void MLUMemcpyH2DSync(void* dst, const void* src, size_t num);

//! Copy memory from address device to device asynchronously in a single device.
void MLUMemcpyD2DAsync(void* dst, const void* src, size_t num,
                       mluStream stream);

//! Copy memory from address device to device synchronously in a single device.
void MLUMemcpyD2DSync(void* dst, const void* src, size_t num);

//! Copy memory from one device to another device asynchronously.
void MLUMemcpyPeerAsync(void* dst, int dst_place, const void* src,
                        int src_place, size_t num, mluStream stream);

//! Copy memory from one device to another device synchronously.
void MLUMemcpyPeerSync(void* dst, int dst_place, const void* src, int src_place,
                       size_t num);

//! Set memory dst with value count size asynchronously
void MLUMemsetAsync(void* dst, int value, size_t count, mluStream stream);

//! Blocks until stream has completed all operations.
void MLUStreamSync(mluStream stream);

//! MLUMalloc with recorded info
cnrtStatus RecordedMLUMalloc(void** ptr, size_t size, int dev_id);

//! MLUFree with recorded info
void RecordedMLUFree(void* p, size_t size, int dev_id);

//! Get available and total mlu memory with considering limitation
bool RecordedMLUMemGetInfo(size_t* avail, size_t* total, size_t* actual_avail,
                           size_t* actual_total, int dev_id);

//! Get recorded mluMalloc size. If record is disabled, return 0.
uint64_t RecordedMLUMallocSize(int dev_id);

bool IsMLUMallocRecorded(int dev_id);

//! Empty idle cached memory held by the allocator.
void EmptyCache(void);

class MLUDeviceGuard {
 public:
  explicit inline MLUDeviceGuard(int dev_id) {
    int prev_id = platform::GetMLUCurrentDeviceId();
    if (prev_id != dev_id) {
      prev_id_ = prev_id;
      platform::SetMLUDeviceId(dev_id);
    }
  }

  inline ~MLUDeviceGuard() {
    if (prev_id_ != -1) {
      platform::SetMLUDeviceId(prev_id_);
    }
  }

  MLUDeviceGuard(const MLUDeviceGuard& o) = delete;
  MLUDeviceGuard& operator=(const MLUDeviceGuard& o) = delete;

 private:
  int prev_id_{-1};
};

}  // namespace platform
}  // namespace paddle

#endif
