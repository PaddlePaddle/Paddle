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
#include "glog/logging.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue_utils.h"

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

void CudaTracer::PrepareTracing() { EnableCuptiActivity(); }

void CudaTracer::StopTracing() { DisableCuptiActivity(); }

int CudaTracer::ProcessCuptiActivity() {
  int record_cnt = 0;
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
  return std::move(buffers);
}

void CudaTracer::ReleaseBuffer(uint8_t* buffer) {
  paddle::framework::AlignedFree(buffer);
}

}  // namespace platform
}  // namespace paddle
