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

#pragma once

#include <memory>
#include <string>

#include "paddle/fluid/platform/device/gcu/runtime/rt_event.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_memory.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_stream.h"

namespace paddle {
namespace distributed {
class ProcessGroupCustom;
}  // namespace distributed
}  // namespace paddle

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

using GcuCtxPtr = std::shared_ptr<paddle::platform::gcu::runtime::Context>;
using GcuMemPtr = std::shared_ptr<paddle::platform::gcu::runtime::Memory>;
using GcuStreamPtr = std::shared_ptr<paddle::platform::gcu::runtime::Stream>;
using GcuEventPtr = std::shared_ptr<paddle::platform::gcu::runtime::Event>;
using GcuRtInfoPtr =
    std::shared_ptr<paddle::platform::gcu::runtime::GcuRunTimeInfo>;
using ProcessGroupCustom = paddle::distributed::ProcessGroupCustom;

void GcuSetRandomSeed(uint64_t seed);

int GcuVisibleDeviceCount();

int GcuGetCurrentDevice();

void GcuSetCurrentDevice(int device_id);

GcuCtxPtr GcuGetContext(int device_id);

bool GcuHasRuntimeInfo(int device_id);

GcuRtInfoPtr GcuGetRuntimeInfo(int device_id);

void GcuSetRuntimeInfo(int device_id, const GcuRtInfoPtr &rt_info);

std::shared_ptr<ProcessGroupCustom> GcuGetProcessGroup(uint32_t group_id = 0);

void GcuSetProcessGroup(uint32_t group_id,
                        const std::shared_ptr<ProcessGroupCustom> &group);

GcuMemPtr GcuTryCreateMemory(GcuCtxPtr ctx, uint64_t nbytes);

GcuMemPtr GcuCreateMemory(GcuCtxPtr ctx, uint64_t nbytes);

GcuMemPtr GcuCreateSubMemory(GcuMemPtr parent, uint64_t offest, uint64_t size);

GcuEventPtr GcuCreateEvent(GcuCtxPtr ctx);

GcuStreamPtr GcuCreateStream(GcuCtxPtr ctx);

GcuStreamPtr GcuGetExeDefaultStream(GcuCtxPtr ctx);

GcuStreamPtr GcuGetDmaDefaultStream(GcuCtxPtr ctx);

void GcuStreamSynchronize(GcuStreamPtr pstream);

void GcuMemcpyH2DSync(GcuMemPtr pmem, const void *pdata, uint64_t nbytes);

void GcuMemcpyD2HSync(void *pdata, GcuMemPtr pmem, uint64_t nbytes);

void GcuMemcpyD2DSync(GcuMemPtr pdst, GcuMemPtr psrc, uint64_t nbytes);

void GcuMemset32Sync(GcuMemPtr pmem, uint32_t pattern, uint64_t nbytes);

void GcuMemcpyH2DAsync(GcuMemPtr pmem,
                       const void *pdata,
                       uint64_t nbytes,
                       GcuStreamPtr pstream);

void GcuMemcpyD2HAsync(void *pdata,
                       GcuMemPtr pmem,
                       uint64_t nbytes,
                       GcuStreamPtr pstream);

void GcuMemcpyD2DAsync(GcuMemPtr pdst,
                       GcuMemPtr psrc,
                       uint64_t nbytes,
                       GcuStreamPtr pstream);

void GcuMemsetAsync(GcuMemPtr pmem,
                    uint32_t pattern,
                    uint64_t nbytes,
                    GcuStreamPtr pstream);

void GcuEventRecord(GcuEventPtr pevent, GcuStreamPtr pstream);

void GcuStreamWaitEvent(GcuStreamPtr pstream, GcuEventPtr pevent);

void GcuStreamWait(GcuStreamPtr pstream_record, GcuStreamPtr pstream_wait);

void GcuLaunchHostFunc(GcuStreamPtr pstream, const std::function<void()> &fn);

std::string GcuGetRTMetricsReport();

void GcuSynchronizeDevice(int device);

void GcuSynchronizeAllContext();

void *GcuDevAddr(GcuMemPtr pmem);

void *GcuStreamImpl(GcuStreamPtr pstream);

topsError_t GcuNativeAlloc(void **ptr, size_t size, int dev_id);

void GcuNativeFree(void *p, size_t size, int dev_id);

void GcuMemcpySync(void *dst,
                   const void *src,
                   size_t size_bytes,
                   topsMemcpyKind kind);

void GcuMemcpyAsync(void *dst,
                    const void *src,
                    size_t size_bytes,
                    topsMemcpyKind kind,
                    topsStream_t stream);

void GcuMemsetAsync(void *dst,
                    int value,
                    size_t size_bytes,
                    topsStream_t stream);

void GcuStreamSynchronize(topsStream_t stream);

void GcuAddStreamCallback(topsStream_t stream,
                          topsStreamCallback_t callback,
                          void *user_data,
                          unsigned int flags = 0);

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
