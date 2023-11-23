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

#include "paddle/fluid/platform/device/gcu/runtime/gcu_rt_interface.h"

#include <gcu/2_0/runtime/platform.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/device/gcu/runtime/rt_resources.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {
namespace {
void CheckMemorys(const std::vector<GcuMemPtr> &ins, Context *ctx) {
  for (auto &pmem : ins) {
    PADDLE_ENFORCE_NOT_NULL(
        pmem, paddle::platform::errors::InvalidArgument("Invalid item of ins"));
    PADDLE_ENFORCE_EQ(pmem->ctx,
                      ctx,
                      platform::errors::InvalidArgument(
                          "The same ctx is required, but get %s vs %s",
                          pmem->ctx->GetName().c_str(),
                          ctx->GetName().c_str()));
  }
}
}  // namespace

void GcuSetRandomSeed(uint64_t seed) {
  // manual seed set based on runtime 2.0
  efrt::Platform::GetInstance().SetRandomSeed(seed);
}

int GcuVisibleDeviceCount() { return Context::VisibleDeviceCount(); }

int GcuGetCurrentDevice() {
  return ResourceMgr::GetInstance()->GetCurrentDevice();
}

void GcuSetCurrentDevice(int device_id) {
  RT_CHECK(topsSetDevice(device_id));
  ResourceMgr::GetInstance()->SetCurrentDevice(device_id);
}

GcuCtxPtr GcuGetContext(int device_id) {
  auto ctx_caches = ResourceMgr::GetInstance()->GetContextCaches();
  auto ctx_cache = ctx_caches->Get(device_id);
  if (ctx_cache == nullptr) {
    ctx_cache = Context::CreateContext(device_id);
    ctx_caches->Add(device_id, ctx_cache);
  }
  return ctx_cache;
}

bool GcuHasRuntimeInfo(int device_id) {
  auto rt_info_caches = ResourceMgr::GetInstance()->GetRuntimeInfoCaches();
  auto rt_info_cache = rt_info_caches->Get(device_id);
  return (rt_info_cache != nullptr);
}

GcuRtInfoPtr GcuGetRuntimeInfo(int device_id) {
  auto rt_info_caches = ResourceMgr::GetInstance()->GetRuntimeInfoCaches();
  auto rt_info_cache = rt_info_caches->Get(device_id);
  PADDLE_ENFORCE_NOT_NULL(
      rt_info_cache,
      paddle::platform::errors::InvalidArgument(
          "Expect runtime info has been created, device id:%u.", device_id));
  return rt_info_cache;
}

void GcuSetRuntimeInfo(int device_id, const GcuRtInfoPtr &rt_info) {
  auto rt_info_caches = ResourceMgr::GetInstance()->GetRuntimeInfoCaches();
  auto rt_info_cache = rt_info_caches->Get(device_id);
  if (rt_info_cache != nullptr) {
    VLOG(1)
        << "[INFO]Runtime info already exist, will be overwritten, device id:"
        << device_id;
    rt_info_caches->Erase(device_id);
  }
  rt_info_caches->Add(device_id, rt_info);
}

std::shared_ptr<ProcessGroupCustom> GcuGetProcessGroup(uint32_t group_id) {
  auto pg_caches = ResourceMgr::GetInstance()->GetProcessGroupCaches();
  auto pg_cache = pg_caches->Get(group_id);
  PADDLE_ENFORCE_NOT_NULL(
      pg_cache,
      paddle::platform::errors::InvalidArgument(
          "Expect process group has been created, group_id id:%u.", group_id));
  return pg_cache;
}

void GcuSetProcessGroup(uint32_t group_id,
                        const std::shared_ptr<ProcessGroupCustom> &group) {
  auto pg_caches = ResourceMgr::GetInstance()->GetProcessGroupCaches();
  auto pg_cache = pg_caches->Get(group_id);
  PADDLE_ENFORCE_EQ(pg_cache,
                    nullptr,
                    platform::errors::InvalidArgument(
                        "Process group already exist, group_id: %u", group_id));
  pg_caches->Add(group_id, group);
}

GcuMemPtr GcuTryCreateMemory(GcuCtxPtr ctx, uint64_t nbytes) {
  PADDLE_ENFORCE_NOT_NULL(
      ctx, paddle::platform::errors::InvalidArgument("Invalid input"));
  return Memory::CreateMemory(ctx.get(), nbytes);
}

GcuMemPtr GcuCreateMemory(GcuCtxPtr ctx, uint64_t nbytes) {
  auto mem = GcuTryCreateMemory(ctx, nbytes);
  PADDLE_ENFORCE_NOT_NULL(mem,
                          paddle::platform::errors::Fatal(
                              "Failed to create memory, nbytes:%lu", nbytes));
  return mem;
}

GcuMemPtr GcuCreateSubMemory(GcuMemPtr parent, uint64_t offest, uint64_t size) {
  PADDLE_ENFORCE_NOT_NULL(
      parent, paddle::platform::errors::InvalidArgument("Invalid input"));
  PADDLE_ENFORCE_NOT_NULL(
      parent->ctx, paddle::platform::errors::InvalidArgument("Invalid parent"));
  return Memory::CreateSubMemory(parent, offest, size);
}

GcuEventPtr GcuCreateEvent(GcuCtxPtr ctx) {
  PADDLE_ENFORCE_NOT_NULL(
      ctx, paddle::platform::errors::InvalidArgument("Invalid input"));
  return Event::CreateNativeEvent(ctx.get());
}

std::shared_ptr<Stream> GcuCreateStream(GcuCtxPtr ctx) {
  PADDLE_ENFORCE_NOT_NULL(
      ctx, paddle::platform::errors::InvalidArgument("Invalid input"));
  return Stream::CreateStream(ctx.get());
}

GcuStreamPtr GcuGetExeDefaultStream(GcuCtxPtr ctx) {
  PADDLE_ENFORCE_NOT_NULL(
      ctx, paddle::platform::errors::InvalidArgument("Invalid input"));
  return ctx->default_exe_stream;
}

GcuStreamPtr GcuGetDmaDefaultStream(GcuCtxPtr ctx) {
  PADDLE_ENFORCE_NOT_NULL(
      ctx, paddle::platform::errors::InvalidArgument("Invalid input"));
  return ctx->default_dma_stream;
}

void GcuStreamSynchronize(GcuStreamPtr pstream) {
  PADDLE_ENFORCE_NOT_NULL(
      pstream, paddle::platform::errors::InvalidArgument("Invalid input"));
  pstream->Synchronize();
}

void GcuMemcpyH2DSync(GcuMemPtr pmem, const void *pdata, uint64_t nbytes) {
  PADDLE_ENFORCE_EQ(
      (pmem && pdata),
      true,
      platform::errors::InvalidArgument(
          "Invalid inputs, pmem is nullptr:%d, pdata is nullptr:%d",
          pmem == nullptr,
          pdata == nullptr));
  auto pstream = pmem->ctx->default_dma_stream;
  pstream->MemcpyH2DSync(pmem, pdata, nbytes);
}

void GcuMemcpyD2HSync(void *pdata, GcuMemPtr pmem, uint64_t nbytes) {
  PADDLE_ENFORCE_EQ(
      (pmem && pdata),
      true,
      platform::errors::InvalidArgument(
          "Invalid inputs, pmem is nullptr:%d, pdata is nullptr:%d",
          pmem == nullptr,
          pdata == nullptr));
  auto pstream = pmem->ctx->default_dma_stream;
  pstream->MemcpyD2HSync(pdata, pmem, nbytes);
}

void GcuMemcpyD2DSync(GcuMemPtr pdst, GcuMemPtr psrc, uint64_t nbytes) {
  PADDLE_ENFORCE_EQ(
      (psrc && pdst),
      true,
      platform::errors::InvalidArgument(
          "Invalid inputs, psrc is nullptr:%d, pdst is nullptr:%d",
          psrc == nullptr,
          pdst == nullptr));
  PADDLE_ENFORCE_EQ(
      psrc->ctx,
      pdst->ctx,
      platform::errors::InvalidArgument("The ctx of src and dst must be "
                                        "the same, but get src: %s vs dst: %s",
                                        psrc->ctx->GetName().c_str(),
                                        pdst->ctx->GetName().c_str()));
  auto pstream = pdst->ctx->default_dma_stream;
  pstream->MemcpyD2DSync(pdst, psrc, nbytes);
}

void GcuMemset32Sync(GcuMemPtr pmem, uint32_t pattern, uint64_t nbytes) {
  PADDLE_ENFORCE_NOT_NULL(
      pmem, paddle::platform::errors::InvalidArgument("Invalid input"));
  auto pstream = pmem->ctx->default_dma_stream;
  pstream->Memset32Sync(pmem, pattern, nbytes);
}

void GcuMemcpyH2DAsync(GcuMemPtr pmem,
                       const void *pdata,
                       uint64_t nbytes,
                       GcuStreamPtr pstream) {
  PADDLE_ENFORCE_EQ((pmem && pdata && pstream),
                    true,
                    platform::errors::InvalidArgument(
                        "Invalid inputs, pmem is nullptr:%d, pdata is "
                        "nullptr:%d, pstream is nullptr:%d",
                        pmem == nullptr,
                        pdata == nullptr,
                        pstream == nullptr));
  PADDLE_ENFORCE_EQ(
      pmem->ctx,
      pstream->ctx,
      platform::errors::InvalidArgument("The ctx of mem and stream must be the "
                                        "same, but get mem: %s vs stream: %s",
                                        pmem->ctx->GetName().c_str(),
                                        pstream->ctx->GetName().c_str()));
  pstream->MemcpyH2DAsync(pmem, pdata, nbytes);
}

void GcuMemcpyD2HAsync(void *pdata,
                       GcuMemPtr pmem,
                       uint64_t nbytes,
                       GcuStreamPtr pstream) {
  PADDLE_ENFORCE_EQ((pmem && pdata && pstream),
                    true,
                    platform::errors::InvalidArgument(
                        "Invalid inputs, pmem is nullptr:%d, pdata is "
                        "nullptr:%d, pstream is nullptr:%d",
                        pmem == nullptr,
                        pdata == nullptr,
                        pstream == nullptr));
  PADDLE_ENFORCE_EQ(
      pmem->ctx,
      pstream->ctx,
      platform::errors::InvalidArgument("The ctx of mem and stream must be the "
                                        "same, but get mem: %s vs stream: %s",
                                        pmem->ctx->GetName().c_str(),
                                        pstream->ctx->GetName().c_str()));
  pstream->MemcpyD2HAsync(pdata, pmem, nbytes);
}

void GcuMemcpyD2DAsync(GcuMemPtr pdst,
                       GcuMemPtr psrc,
                       uint64_t nbytes,
                       GcuStreamPtr pstream) {
  PADDLE_ENFORCE_EQ((pdst && psrc && pstream),
                    true,
                    platform::errors::InvalidArgument(
                        "Invalid inputs, psrc is nullptr:%d, pdst is "
                        "nullptr:%d, pstream is nullptr:%d",
                        psrc == nullptr,
                        pdst == nullptr,
                        pstream == nullptr));
  PADDLE_ENFORCE_EQ(
      (pdst->ctx == psrc->ctx && pdst->ctx == pstream->ctx),
      true,
      platform::errors::InvalidArgument(
          "The ctx of src, dst and stream must be the same, but get src: %s, "
          "dst:%s, stream: %s",
          psrc->ctx->GetName().c_str(),
          pdst->ctx->GetName().c_str(),
          pstream->ctx->GetName().c_str()));
  pstream->MemcpyD2DAsync(pdst, psrc, nbytes);
}

void GcuMemsetAsync(GcuMemPtr pmem,
                    uint32_t pattern,
                    uint64_t nbytes,
                    GcuStreamPtr pstream) {
  PADDLE_ENFORCE_EQ(
      (pmem && pstream),
      true,
      platform::errors::InvalidArgument(
          "Invalid inputs, pmem is nullptr:%d, pstream is nullptr:%d",
          pmem == nullptr,
          pstream == nullptr));
  PADDLE_ENFORCE_EQ(
      pmem->ctx,
      pstream->ctx,
      platform::errors::InvalidArgument("The ctx of mem and stream must be the "
                                        "same, but get mem: %s vs stream: %s",
                                        pmem->ctx->GetName().c_str(),
                                        pstream->ctx->GetName().c_str()));
  pstream->MemsetAsync(pmem, pattern, nbytes);
}

void GcuEventRecord(GcuEventPtr pevent, GcuStreamPtr pstream) {
  PADDLE_ENFORCE_EQ(
      (pevent && pstream),
      true,
      platform::errors::InvalidArgument(
          "Invalid inputs, pevent is nullptr:%d, pstream is nullptr:%d",
          pevent == nullptr,
          pstream == nullptr));
  pstream->EventRecord(pevent);
}

void GcuStreamWaitEvent(GcuStreamPtr pstream, GcuEventPtr pevent) {
  PADDLE_ENFORCE_EQ(
      (pevent && pstream),
      true,
      platform::errors::InvalidArgument(
          "Invalid inputs, pevent is nullptr:%d, pstream is nullptr:%d",
          pevent == nullptr,
          pstream == nullptr));
  pstream->WaitEvent(pevent);
}

void GcuStreamWait(GcuStreamPtr pstream_record, GcuStreamPtr pstream_wait) {
  PADDLE_ENFORCE_EQ((pstream_record && pstream_wait),
                    true,
                    platform::errors::InvalidArgument(
                        "Invalid inputs, pstream_record is nullptr:%d, "
                        "pstream_wait is nullptr:%d",
                        pstream_record == nullptr,
                        pstream_wait == nullptr));
  if (pstream_record.get() == pstream_wait.get()) {
    return;
  }
  PADDLE_ENFORCE_EQ(
      pstream_record->ctx,
      pstream_wait->ctx,
      platform::errors::InvalidArgument(
          "The ctx of pstream_record and pstream_wait must be "
          "the same, but get pstream_record: %s vs pstream_wait: %s",
          pstream_record->ctx->GetName().c_str(),
          pstream_wait->ctx->GetName().c_str()));
  auto pevent = Event::CreateNativeEvent(pstream_record->ctx);
  GcuEventRecord(pevent, pstream_record);
  GcuStreamWaitEvent(pstream_wait, pevent);
  auto fn = [pevent]() -> void {
    /* do nothing, just extend the life cycle of pevent */
  };
  GcuLaunchHostFunc(pstream_wait, std::move(fn));
}

void GcuLaunchHostFunc(GcuStreamPtr pstream, const std::function<void()> &fn) {
  PADDLE_ENFORCE_NOT_NULL(
      pstream, paddle::platform::errors::InvalidArgument("Invalid input"));
  pstream->AddHostCallback(std::move(fn));
}

std::string GcuGetRTMetricsReport() {
  return ResourceMgr::GetInstance()->GetRTMetricsReport();
}

void GcuSynchronizeDevice(int device) {
  auto ctx = GcuGetContext(device);
  PADDLE_ENFORCE_NOT_NULL(ctx,
                          paddle::platform::errors::Unavailable(
                              "Failed to get context, device id:%d", device));
  ctx->Synchronize();
}

void GcuSynchronizeAllContext() {
  uint32_t ndevs = GcuVisibleDeviceCount();
  auto ctx_caches = ResourceMgr::GetInstance()->GetContextCaches();
  for (uint32_t did = 0; did < ndevs; did++) {
    auto ctx_cache = ctx_caches->Get(did);
    if (ctx_cache) {
      ctx_cache->Synchronize();
    }
  }
}

void *GcuDevAddr(GcuMemPtr pmem) {
  PADDLE_ENFORCE_NOT_NULL(
      pmem, paddle::platform::errors::InvalidArgument("Invalid input"));
  PADDLE_ENFORCE_EQ(pmem->IsValid(),
                    true,
                    paddle::platform::errors::InvalidArgument("Invalid mem"));
  return pmem->GetDevAddr();
}

void *GcuStreamImpl(GcuStreamPtr pstream) {
  PADDLE_ENFORCE_EQ((pstream && pstream->tops_stream),
                    true,
                    platform::errors::InvalidArgument(
                        "Invalid inputs, pstream is nullptr:%d, "
                        "pstream->tops_stream is nullptr:%d",
                        pstream == nullptr,
                        pstream->tops_stream == nullptr));
  return pstream->tops_stream;
}

topsError_t GcuNativeAlloc(void **ptr, size_t size, int device) {
  GcuDeviceGuard guard(device);
  VLOG(10) << "[GcuNativeAlloc] size = " << size;
  auto ret = topsMalloc(ptr, size);
  if (ret == topsSuccess) {
    ResourceMgr::GetInstance()->RTCounter(device + "_NativeMemory", 1);
    ResourceMgr::GetInstance()->RTCounter(device + "_NativeMemoryUse", size);
  }
  return ret;
}

void GcuNativeFree(void *p, size_t size, int device) {
  GcuDeviceGuard guard(device);
  VLOG(10) << "[GcuNativeFree] size = " << size;
  RT_CHECK(topsFree(p));
  ResourceMgr::GetInstance()->RTCounter(device + "_NativeMemory", -1);
  ResourceMgr::GetInstance()->RTCounter(device + "_NativeMemoryUse", -size);
}

void GcuMemcpySync(void *dst,
                   const void *src,
                   size_t size_bytes,
                   topsMemcpyKind kind) {
  GcuDeviceGuard guard(GcuGetCurrentDevice());
  RT_CHECK(topsMemcpy(dst, src, size_bytes, kind));
  ResourceMgr::GetInstance()->RTCounter("MemcpySyncWithKind", 1);
}

void GcuMemcpyAsync(void *dst,
                    const void *src,
                    size_t size_bytes,
                    topsMemcpyKind kind,
                    topsStream_t stream) {
  GcuDeviceGuard guard(GcuGetCurrentDevice());
  RT_CHECK(topsMemcpyAsync(dst, src, size_bytes, kind, stream));
  ResourceMgr::GetInstance()->RTCounter("MemcpyAsyncWithKind", 1);
}

void GcuMemsetAsync(void *dst,
                    int value,
                    size_t size_bytes,
                    topsStream_t stream) {
  GcuDeviceGuard guard(GcuGetCurrentDevice());
  RT_CHECK(topsMemsetAsync(dst, value, size_bytes, stream));
  ResourceMgr::GetInstance()->RTCounter("MemsetAsync", 1);
}

void GcuStreamSynchronize(topsStream_t stream) {
  GcuDeviceGuard guard(GcuGetCurrentDevice());
  RT_CHECK(topsStreamSynchronize(stream));
  ResourceMgr::GetInstance()->RTCounter("Synchronize", 1);
}

void GcuAddStreamCallback(topsStream_t stream,
                          topsStreamCallback_t callback,
                          void *user_data,
                          unsigned int flags) {
  GcuDeviceGuard guard(GcuGetCurrentDevice());
  RT_CHECK(topsStreamAddCallback(stream, callback, user_data, flags));
  ResourceMgr::GetInstance()->RTCounter("AddStreamCallback", 1);
}

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
