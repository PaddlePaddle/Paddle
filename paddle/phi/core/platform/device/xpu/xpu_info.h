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

#pragma once

#ifdef PADDLE_WITH_XPU
#include <vector>

#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/place.h"
#include "xpu/runtime.h"

namespace paddle {

using xpuStream = XPUStream;
using xpuEventHandle = XPUEvent;

namespace platform {

/***** Version Management *****/

//! Get the version of XPU Driver
int GetDriverVersion();

//! Get the version of XPU Runtime
int GetRuntimeVersion();

/***** Device Management *****/

//! Get the total number of XPU devices in system.
int GetXPUDeviceCount();

//! Set the XPU device id for next execution.
void SetXPUDeviceId(int device_id);

//! Get the current XPU device id in system.
int GetXPUCurrentDeviceId();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices();

/***** Memory Management *****/

//! Copy memory from address src to dst synchronously.
void MemcpySyncH2D(void *dst,
                   const void *src,
                   size_t count,
                   const phi::XPUPlace &dst_place);
void MemcpySyncD2H(void *dst,
                   const void *src,
                   size_t count,
                   const phi::XPUPlace &src_place);
void MemcpySyncD2D(void *dst,
                   const phi::XPUPlace &dst_place,
                   const void *src,
                   const phi::XPUPlace &src_place,
                   size_t count);

//! Blocks until stream has completed all operations.
void XPUStreamSync(xpuStream stream);

using XPUDeviceGuard = phi::backends::xpu::XPUDeviceGuard;

phi::backends::xpu::XPUVersion get_xpu_version(int dev_id);

void set_xpu_debug_level(int level);

//! Get the minimum chunk size for XPU allocator.
size_t XPUMinChunkSize();

//! xpu_malloc with recorded info
int RecordedXPUMalloc(void **ptr, size_t size, int dev_id);

//! xpu_free with recorded info
void RecordedXPUFree(void *p, size_t size, int dev_id);

//! Get recorded actrtMalloc size. If record is disabled, return 0.
uint64_t RecordedXPUMallocSize(int dev_id);

uint64_t RecordedXPULimitSize(int dev_id);

bool IsXPUMallocRecorded(int dev_id);

void EmptyCache(void);

}  // namespace platform
}  // namespace paddle
#endif
