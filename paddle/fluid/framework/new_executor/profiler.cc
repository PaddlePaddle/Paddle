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

#include "paddle/fluid/framework/new_executor/profiler.h"
#include <glog/logging.h>

namespace paddle {
namespace framework {
namespace profiler {

OpDeviceProfileEvent::OpDeviceProfileEvent(
    const DeviceContext& device_context) {
#if defined(PADDLE_WITH_CUDA)
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&this->event_obj_cuda_));
  auto* gpu_context = dynamic_cast<const phi::GPUContext*>(device_context);
  if (gpu_context) {
    this->cuda_stream_ = gpu_context->stream();
  } else {
    VLOG(1) << "Cannot obtain cuda stream from device context. Use default "
               "cuda stream instead. "
               "This may cause error during profiling!";
    this->cuda_stream_ = nullptr;
  }
#endif
}

OpDeviceProfileEvent::~OpDeviceProfileEvent() {
#if defined(PADDLE_WITH_CUDA)
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(this->event_obj_cuda_));
#endif
}

void OpDeviceProfileEvent::Record() {
#if defined(PADDLE_WITH_CUDA)
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventRecord(this->event_obj_cuda_, this->cuda_stream_));
#endif
  platform::Timer timer;
  timeval _now;
  gettimeofday(&_now, nullptr);
  this->event_obj_cpu_.event_time_us_ =
      _now.tv_sec * 1000 * 1000 + _now.tv_usec;
}

OpRuntimeProfiler::OpRuntimeProfiler(const DeviceContext& device_context)
    : device_context_(device_context) {}

OpRuntimeProfiler::~OpRuntimeProfiler() {}

void OpRuntimeProfiler::RecordEvent(const std::string& event_name) {
  if (this->name_to_device_profile_events_.find(event_name) !=
      this->name_to_device_profile_events_.end()) {
    VLOG(1) << "An event with name \"" << event_name
            << "\" already exists. "
               "RecordEvent will overwrite this event."
  }
  // create a new event
  std::shared_ptr event_ptr =
      std::make_shared<OpDeviceProfileEvent>(device_context_);
  event_ptr->Record();
  this->name_to_device_profile_events_.insert_or_assign(event_name, event_ptr);
}

std::tuple<double, double> OpRuntimeProfiler::MeasureTimeLapseBetweenEvents(
    const OpDeviceProfileEvent& event_start,
    const OpDeviceProfileEvent& event_end) const {}

}  // namespace profiler
}  // namespace framework
}  // namespace paddle
