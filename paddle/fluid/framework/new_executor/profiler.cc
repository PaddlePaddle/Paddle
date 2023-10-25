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

namespace paddle {
namespace framework {
namespace profiler {

OpDeviceProfileEvent::OpDeviceProfileEvent(
    const platform::DeviceContext& device_context) {
#if defined(PADDLE_WITH_CUDA)
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&this->event_obj_cuda_));
  auto* gpu_context = dynamic_cast<const phi::GPUContext*>(&device_context);
  if (gpu_context) {
    this->cuda_stream_ = gpu_context->stream();
    VLOG(4) << "Successfully obtained cuda stream from device context.";
  } else {
    VLOG(1) << "Cannot obtain cuda stream from device context. Use default "
               "cuda stream (null stream) instead. "
               "This may cause error during profiling!";
    this->cuda_stream_ = nullptr;
  }
#endif
}

OpDeviceProfileEvent::~OpDeviceProfileEvent() {
#if defined(PADDLE_WITH_CUDA)
  // Cannot use PADDLE_ENFORCE_GPU_SUCCESS here since it
  // will cause "throw() will always call terminate()"
  // warning and warning will be treated as error. So i
  // directly call CUDA API here.
  if (this->event_obj_cuda_) {
    cudaEventDestroy(this->event_obj_cuda_);
  }
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

std::tuple<double, double> OpDeviceProfileEvent::MeasureTimeLapseWrtOtherEvent(
    const OpDeviceProfileEvent& end_event) const {
  double device_time_lapse_us = -1.0;
#if defined(PADDLE_WITH_CUDA)
  if (cudaEventQuery(this->event_obj_cuda_) != cudaSuccess ||
      cudaEventQuery(end_event.event_obj_cuda_) != cudaSuccess) {
    VLOG(1) << "cuda event not set yet, time lapse measurment will return an "
               "invalid value.";
    return std::make_tuple(-1.0, -1.0);
  } else {
    float dt_ms = -1.0f;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventElapsedTime(
        &dt_ms, this->event_obj_cuda_, end_event.event_obj_cuda_));
    // cuda measures time lapse in ms, convert it to us
    device_time_lapse_us = static_cast<double>(dt_ms) * 1000.0;
    VLOG(4) << "measured device time lapse: " << device_time_lapse_us << " us.";
  }
#else
  // implement other platforms
#endif
  double cpu_time_lapse_us =
      static_cast<double>(end_event.event_obj_cpu_.event_time_us_ -
                          this->event_obj_cpu_.event_time_us_);
  return std::make_tuple(cpu_time_lapse_us, device_time_lapse_us);
}

OpRuntimeProfiler::OpRuntimeProfiler() {}

OpRuntimeProfiler::~OpRuntimeProfiler() {}

void OpRuntimeProfiler::RecordEvent(
    const std::string& event_name,
    const platform::DeviceContext& device_context) {
  if (this->name_to_device_profile_events_.find(event_name) !=
      this->name_to_device_profile_events_.end()) {
    VLOG(1) << "An event with name \"" << event_name
            << "\" already exists. "
               "RecordEvent will overwrite this event.";
  }
  // create a new event
  std::shared_ptr event_ptr =
      std::make_shared<OpDeviceProfileEvent>(device_context);
  event_ptr->Record();
  this->name_to_device_profile_events_.insert_or_assign(event_name, event_ptr);
}

std::tuple<double, double> OpRuntimeProfiler::MeasureTimeLapseBetweenEvents(
    const std::string& start_event_name,
    const std::string& end_event_name) const {
  if (this->name_to_device_profile_events_.find(start_event_name) ==
          this->name_to_device_profile_events_.end() ||
      this->name_to_device_profile_events_.find(end_event_name) ==
          this->name_to_device_profile_events_.end()) {
    VLOG(1) << "One of the event name is not found. " << start_event_name << " "
            << end_event_name;
    return std::make_tuple(-1.0, -1.0);
  }
  std::shared_ptr<OpDeviceProfileEvent> start_event =
      this->name_to_device_profile_events_.at(start_event_name);
  std::shared_ptr<OpDeviceProfileEvent> end_event =
      this->name_to_device_profile_events_.at(start_event_name);
  return start_event->MeasureTimeLapseWrtOtherEvent(*end_event);
}

}  // namespace profiler
}  // namespace framework
}  // namespace paddle
