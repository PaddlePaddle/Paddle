// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/profiler/xpu_tracer.h"

#include <mutex>
#include <unordered_map>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/os_info.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/device_manager.h"
#endif

#define XPTI_CALL(call)                                                       \
  do {                                                                        \
    XPTIResult _status = call;                                                \
    if (_status != XPTI_SUCCESS) {                                            \
      LOG(ERROR) << "Function " << #call << " failed with error " << _status; \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)

namespace paddle {
namespace platform {

void XPUTracer::PrepareTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::UNINITED || state_ == TracerState::STOPED,
      true,
      platform::errors::PreconditionNotMet("XPUTracer must be UNINITED"));
#ifdef PADDLE_WITH_XPU
  XPTI_CALL(dynload::xptiActivityEnable());
  VLOG(3) << "enable xpti activity";
#endif
  state_ = TracerState::READY;
}

void XPUTracer::StartTracing() {
  PADDLE_ENFORCE_EQ(
      state_ == TracerState::READY,
      true,
      platform::errors::PreconditionNotMet("Tracer must be READY or STOPPED"));
#ifdef PADDLE_WITH_XPU
  XPTI_CALL(dynload::xptiStartTracing());
#endif
  tracing_start_ns_ = PosixInNsec();
  state_ = TracerState::STARTED;
}

void XPUTracer::StopTracing() {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STARTED,
      platform::errors::PreconditionNotMet("Tracer must be STARTED"));
#ifdef PADDLE_WITH_XPU
  XPTI_CALL(dynload::xptiStopTracing());
  XPTI_CALL(dynload::xptiActivityDisable());
  VLOG(3) << "disable xpti activity";
#endif
  state_ = TracerState::STOPED;
}

#ifdef PADDLE_WITH_XPU
void AddKernelRecord(const baidu::xpti::XPTIEvent& e,
                     uint64_t start_ns,
                     TraceEventCollector* collector) {
  const baidu::xpti::XPTIEventKernel& kernel = e.kernel;
  if (e.start < start_ns) {
    return;
  }
  DeviceTraceEvent event;
  event.name = demangle(e.name);
  event.type = TracerEventType::Kernel;
  event.start_ns = e.start;
  event.end_ns = e.end;
  event.device_id = kernel.device_id;
  event.stream_id = kernel.stream_id;

  collector->AddDeviceEvent(std::move(event));
}
#endif

void XPUTracer::CollectTraceData(TraceEventCollector* collector) {
  PADDLE_ENFORCE_EQ(
      state_,
      TracerState::STOPED,
      platform::errors::PreconditionNotMet("Tracer must be STOPED"));
#ifdef PADDLE_WITH_XPU
  XPTI_CALL(dynload::xptiActivityFlushAll());
#endif
  baidu::xpti::XPTIEvent e;
  while (true) {
    XPTIResult status = dynload::xptiActivityGetNextRecord(&e);
    if (status == XPTI_SUCCESS) {
      switch (e.type) {
        case XPTI_EVENT_TYPE_KERNEL:
          AddKernelRecord(e, tracing_start_ns_, collector);
          break;
        default:
          break;
      }
    } else if (status == XPTI_INVALID_DATA) {
      // data queue already empty
      break;
    } else {
      XPTI_CALL(status);
    }
  }
}

}  // namespace platform
}  // namespace paddle
