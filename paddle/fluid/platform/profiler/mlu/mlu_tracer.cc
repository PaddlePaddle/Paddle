// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/profiler/mlu/mlu_tracer.h"
#include <string>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue_utils.h"
#include "paddle/fluid/platform/os_info.h"
#include "paddle/fluid/platform/profiler/mlu/cnpapi_data_process.h"

#define CNPAPI_CALL(call)                                                    \
  do {                                                                       \
    cnpapiResult _status = call;                                             \
    if (_status != CNPAPI_SUCCESS) {                                         \
      const char* errstr;                                                    \
      cnpapiGetResultString(_status, &errstr);                               \
      LOG(ERROR) << "Function " << #call << " failed with error " << errstr; \
    }                                                                        \
  } while (0)

namespace paddle {
namespace platform {

namespace {

void BufferRequestedCallback(uint64_t** buffer, size_t* size,
                             size_t* max_num_records) {
  constexpr size_t kBufferSize = 1 << 23;  // 8 MB
  constexpr size_t kBufferAlignSize = 8;
  *buffer = reinterpret_cast<uint64_t*>(
      paddle::framework::AlignedMalloc(kBufferSize, kBufferAlignSize));
  *size = kBufferSize;
  *max_num_records = 0;
}

void BufferCompletedCallback(uint64_t* buffer, size_t size, size_t valid_size) {
  if (buffer == nullptr || valid_size == 0) {
    return;
  }
  auto mlu_tracer = &MluTracer::GetInstance();
  mlu_tracer->ProcessCnpapiActivity(buffer, valid_size);

  paddle::framework::AlignedFree(buffer);
}

}  // namespace

MluTracer::MluTracer() {
#ifdef PADDLE_WITH_MLU
  CNPAPI_CALL(cnpapiInit());
  CNPAPI_CALL(cnpapiActivityRegisterCallbacks(BufferRequestedCallback,
                                              BufferCompletedCallback));
#endif
}

void MluTracer::PrepareTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::UNINITED || state_ == TracerState::STOPED, true,
      platform::errors::PreconditionNotMet("MluTracer must be UNINITED"));
  EnableCnpapiActivity();
  state_ = TracerState::READY;
}

void MluTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(state_ == TracerState::READY, true,
                    platform::errors::PreconditionNotMet(
                        "MluTracer must be READY or STOPPED"));
  tracing_start_ns_ = PosixInNsec();
  state_ = TracerState::STARTED;
}

void MluTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STARTED,
      platform::errors::PreconditionNotMet("MluTracer must be STARTED"));
  DisableCnpapiActivity();
  state_ = TracerState::STOPED;
}

void MluTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_, TracerState::STOPED,
      platform::errors::PreconditionNotMet("MluTracer must be STOPED"));
  for (auto he : collector_.HostEvents()) {
    collector->AddHostEvent(std::move(he));
  }
  for (auto rte : collector_.RuntimeEvents()) {
    collector->AddRuntimeEvent(std::move(rte));
  }
  for (auto de : collector_.DeviceEvents()) {
    collector->AddDeviceEvent(std::move(de));
  }
  for (auto tn : collector_.ThreadNames()) {
    collector->AddThreadName(tn.first, tn.second);
  }
  collector_.ClearAll();
}

void MluTracer::ProcessCnpapiActivity(uint64_t* buffer, size_t valid_size) {
#ifdef PADDLE_WITH_MLU
  cnpapiActivity* record = nullptr;
  while (true) {
    cnpapiResult status =
        cnpapiActivityGetNextRecord(buffer, valid_size, &record);
    if (status == CNPAPI_SUCCESS) {
      details::ProcessCnpapiActivityRecord(record, tracing_start_ns_,
                                           &collector_);
    } else if (status == CNPAPI_ERROR_INSUFFICIENT_MEMORY ||
               status == CNPAPI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      CNPAPI_CALL(status);
    }
  }
#endif
}

void MluTracer::EnableCnpapiActivity() {
#ifdef PADDLE_WITH_MLU
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL));
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMSET));
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_CNDRV_API));
  VLOG(3) << "enable cnpapi activity";
#endif
}

void MluTracer::DisableCnpapiActivity() {
#ifdef PADDLE_WITH_MLU
  CNPAPI_CALL(cnpapiActivityFlushAll());
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_KERNEL));
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMSET));
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_CNDRV_API));
  VLOG(3) << "disable cnpapi activity";
#endif
}

}  // namespace platform
}  // namespace paddle
