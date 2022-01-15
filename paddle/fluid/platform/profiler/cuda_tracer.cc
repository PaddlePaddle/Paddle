/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/profiler/cuda_tracer.h"
#include <string>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue_utils.h"
#include "paddle/fluid/platform/os_info.h"

#define CUPTI_CALL(call)                                                     \
  do {                                                                       \
    CUptiResult _status = call;                                              \
    if (_status != CUPTI_SUCCESS) {                                          \
      const char* errstr;                                                    \
      dynload::cuptiGetResultString(_status, &errstr);                       \
      LOG(ERROR) << "Function " << #call << " failed with error " << errstr; \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)

namespace paddle {
namespace platform {

#ifdef PADDLE_WITH_CUPTI
namespace {

void AddKernelRecord(const CUpti_ActivityKernel4* kernel, uint64_t start_ns,
                     TraceEventCollector* collector) {
  if (kernel->start < start_ns) {
    return;
  }
  DeviceTraceEvent event;
  event.name = kernel->name;
  event.start_ns = kernel->start;
  event.end_ns = kernel->end;
  event.correlation_id = kernel->correlationId;
  collector->AddDeviceEvent(std::move(event));
}

std::string MemcpyKind(uint8_t kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "MEMCPY_HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "MEMCPY_DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "MEMCPY_HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "MEMCPY_AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "MEMCPY_AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "MEMCPY_AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "MEMCPY_DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "MEMCPY_DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "MEMCPY_HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "MEMCPY_PtoP";
    default:
      return "MEMCPY";
      break;
  }
}

void AddMemcpyRecord(const CUpti_ActivityMemcpy* memcpy, uint64_t start_ns,
                     TraceEventCollector* collector) {
  if (memcpy->start < start_ns) {
    return;
  }
  DeviceTraceEvent event;
  event.name = MemcpyKind(memcpy->copyKind);
  event.start_ns = memcpy->start;
  event.end_ns = memcpy->end;
  event.correlation_id = memcpy->correlationId;
  collector->AddDeviceEvent(std::move(event));
}

void AddMemcpy2Record(const CUpti_ActivityMemcpy2* memcpy2, uint64_t start_ns,
                      TraceEventCollector* collector) {
  if (memcpy2->start < start_ns) {
    return;
  }
  DeviceTraceEvent event;
  event.name = MemcpyKind(memcpy2->copyKind);
  event.start_ns = memcpy2->start;
  event.end_ns = memcpy2->end;
  event.correlation_id = memcpy2->correlationId;
  collector->AddDeviceEvent(std::move(event));
}

void AddMemsetRecord(const CUpti_ActivityMemset* memset, uint64_t start_ns,
                     TraceEventCollector* collector) {
  if (memset->start < start_ns) {
    return;
  }
  DeviceTraceEvent event;
  event.name = "MEMSET";
  event.start_ns = memset->start;
  event.end_ns = memset->end;
  event.correlation_id = memset->correlationId;
  collector->AddDeviceEvent(std::move(event));
}

std::unordered_map<CUpti_CallbackId, std::string> runtime_cbid_str;

void InitCuptiRuntimeCbidStr() {
  static bool called = false;
  if (called) return;
  called = true;
#define REGISTER_RUNTIME_CBID_STR(cbid) \
  runtime_cbid_str[CUPTI_RUNTIME_TRACE_CBID_##cbid] = #cbid

  REGISTER_RUNTIME_CBID_STR(cudaBindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaConfigureCall_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetAttribute_v5000);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetStreamPriorityRange_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDriverGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventCreateWithFlags_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventQuery_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventRecord_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFreeHost_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFree_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFuncGetAttributes_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDeviceCount_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDeviceProperties_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetErrorString_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaHostAlloc_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaHostGetDevicePointer_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaLaunchKernel_v7000);
  REGISTER_RUNTIME_CBID_STR(cudaMallocHost_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMalloc_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemsetAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemset_v3020);
  REGISTER_RUNTIME_CBID_STR(
      cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000);
  REGISTER_RUNTIME_CBID_STR(cudaPeekAtLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaRuntimeGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaSetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreate_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreateWithFlags_v5000);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreateWithPriority_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaStreamDestroy_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaStreamSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamWaitEvent_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaUnbindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaSetupArgument_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaLaunch_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetPCIBusId_v4010);
#if CUDA_VERSION >= 9000
  REGISTER_RUNTIME_CBID_STR(cudaLaunchCooperativeKernel_v9000);
  REGISTER_RUNTIME_CBID_STR(cudaLaunchCooperativeKernelMultiDevice_v9000);
#endif

#undef REGISTER_RUNTIME_CBID_STR
}

std::string RuntimeKind(CUpti_CallbackId cbid) {
  auto iter = runtime_cbid_str.find(cbid);
  if (iter == runtime_cbid_str.end())
    return "Runtime API " + std::to_string(cbid);
  return iter->second;
}

std::unordered_map<uint32_t, uint64_t> CreateThreadIdMapping() {
  std::unordered_map<uint32_t, uint64_t> mapping;
  std::unordered_map<uint64_t, ThreadId> ids = GetAllThreadIds();
  for (const auto& id : ids) {
    mapping[id.second.cupti_tid] = id.second.sys_tid;
  }
  return mapping;
}

void AddApiRecord(const CUpti_ActivityAPI* api, uint64_t start_ns,
                  const std::unordered_map<uint32_t, uint64_t> tid_mapping,
                  TraceEventCollector* collector) {
  if (api->start < start_ns) {
    return;
  }
  RuntimeTraceEvent event;
  event.name = RuntimeKind(api->cbid);
  event.start_ns = api->start;
  event.end_ns = api->end;
  uint64_t tid = 0;
  auto iter = tid_mapping.find(api->threadId);
  if (iter == tid_mapping.end()) {
  } else {
    tid = iter->second;
  }
  event.thread_id = tid;
  event.correlation_id = api->correlationId;
  collector->AddRuntimeEvent(std::move(event));
}

void ProcessCuptiActivityRecord(
    const CUpti_Activity* record, uint64_t start_ns,
    const std::unordered_map<uint32_t, uint64_t> tid_mapping,
    TraceEventCollector* collector) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      AddKernelRecord(reinterpret_cast<const CUpti_ActivityKernel4*>(record),
                      start_ns, collector);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      AddMemcpyRecord(reinterpret_cast<const CUpti_ActivityMemcpy*>(record),
                      start_ns, collector);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY2:
      AddMemcpy2Record(reinterpret_cast<const CUpti_ActivityMemcpy2*>(record),
                       start_ns, collector);
      break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
      AddMemsetRecord(reinterpret_cast<const CUpti_ActivityMemset*>(record),
                      start_ns, collector);
      break;
    case CUPTI_ACTIVITY_KIND_DRIVER:
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      AddApiRecord(reinterpret_cast<const CUpti_ActivityAPI*>(record), start_ns,
                   tid_mapping, collector);
      break;
    default:
      break;
  }
}

}  // namespace
#endif

CudaTracer::CudaTracer() {
#ifdef PADDLE_WITH_CUPTI
  InitCuptiRuntimeCbidStr();
#endif
}

void CudaTracer::PrepareTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::UNINITED || state_ == TracerState::STOPED, true,
      platform::errors::PreconditionNotMet("Tracer must be UNINITED"));
  EnableCuptiActivity();
  state_ = TracerState::READY;
}

void CudaTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::READY, true,
      platform::errors::PreconditionNotMet("Tracer must be READY or STOPPED"));
  ConsumeBuffers();
  tracing_start_ns_ = PosixInNsec();
  state_ = TracerState::STARTED;
}

void CudaTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STARTED,
      platform::errors::PreconditionNotMet("Tracer must be STARTED"));
  DisableCuptiActivity();
  state_ = TracerState::STOPED;
}

void CudaTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STOPED,
      platform::errors::PreconditionNotMet("Tracer must be STOPED"));
  ProcessCuptiActivity(collector);
}

int CudaTracer::ProcessCuptiActivity(TraceEventCollector* collector) {
  int record_cnt = 0;
#ifdef PADDLE_WITH_CUPTI
  CUPTI_CALL(dynload::cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  auto mapping = CreateThreadIdMapping();
  std::vector<ActivityBuffer> buffers = ConsumeBuffers();
  for (auto& buffer : buffers) {
    if (buffer.addr == nullptr || buffer.valid_size == 0) {
      continue;
    }

    CUpti_Activity* record = nullptr;
    while (true) {
      CUptiResult status = dynload::cuptiActivityGetNextRecord(
          buffer.addr, buffer.valid_size, &record);
      if (status == CUPTI_SUCCESS) {
        ProcessCuptiActivityRecord(record, tracing_start_ns_, mapping,
                                   collector);
        ++record_cnt;
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        CUPTI_CALL(status);
      }
    }

    ReleaseBuffer(buffer.addr);
  }
#endif
  return record_cnt;
}

void CudaTracer::EnableCuptiActivity() {
#ifdef PADDLE_WITH_CUPTI
  CUPTI_CALL(dynload::cuptiActivityRegisterCallbacks(BufferRequestedCallback,
                                                     BufferCompletedCallback));

  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(
      dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
#endif
}

void CudaTracer::DisableCuptiActivity() {
#ifdef PADDLE_WITH_CUPTI
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(
      dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
#endif
}

#ifdef PADDLE_WITH_CUPTI
void CUPTIAPI CudaTracer::BufferRequestedCallback(uint8_t** buffer,
                                                  size_t* size,
                                                  size_t* max_num_records) {
  GetInstance().AllocateBuffer(buffer, size);
  *max_num_records = 0;
}

void CUPTIAPI CudaTracer::BufferCompletedCallback(CUcontext ctx,
                                                  uint32_t stream_id,
                                                  uint8_t* buffer, size_t size,
                                                  size_t valid_size) {
  GetInstance().ProduceBuffer(buffer, valid_size);
  size_t dropped = 0;
  CUPTI_CALL(
      dynload::cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped));
  if (dropped != 0) {
    LOG(WARNING) << "Stream " << stream_id << " Dropped " << dropped
                 << " activity records";
  }
}
#endif

void CudaTracer::AllocateBuffer(uint8_t** buffer, size_t* size) {
  constexpr size_t kBufSize = 1 << 23;  // 8 MB
  constexpr size_t kBufAlign = 8;       // 8 B
  *buffer = reinterpret_cast<uint8_t*>(
      paddle::framework::AlignedMalloc(kBufSize, kBufAlign));
  *size = kBufSize;
}

void CudaTracer::ProduceBuffer(uint8_t* buffer, size_t valid_size) {
  std::lock_guard<std::mutex> guard(activity_buffer_lock_);
  activity_buffers_.emplace_back(buffer, valid_size);
}

std::vector<CudaTracer::ActivityBuffer> CudaTracer::ConsumeBuffers() {
  std::vector<ActivityBuffer> buffers;
  {
    std::lock_guard<std::mutex> guard(activity_buffer_lock_);
    buffers.swap(activity_buffers_);
  }
  return buffers;
}

void CudaTracer::ReleaseBuffer(uint8_t* buffer) {
  paddle::framework::AlignedFree(buffer);
}

}  // namespace platform
}  // namespace paddle
