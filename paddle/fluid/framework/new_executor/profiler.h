// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <glog/logging.h>
#include <stdlib.h>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace framework {
namespace interpreter {
struct CostInfo {
  double total_time{0.};          // ms
  size_t device_memory_bytes{0};  // total allocated memory size
};

class ProfilerGuard {
 public:
  ProfilerGuard(const platform::Place& place, CostInfo* cost_info)
      : place_(place), cost_info_(cost_info) {
    timer_.Start();
  }

  ~ProfilerGuard() {
    timer_.Pause();
    cost_info_->total_time += timer_.ElapsedMS();
    TotalCUDAAllocatedMemorySize(place_);
  }

 private:
  void TotalCUDAAllocatedMemorySize(const platform::Place& place) {
    if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      auto cuda_place = place;
      cost_info_->device_memory_bytes =
          platform::RecordedGpuMallocSize(cuda_place.device);
#endif
    }
  }

  const platform::Place& place_;
  CostInfo* cost_info_;
  platform::Timer timer_;
};

}  // namespace interpreter

namespace profiler {

class OpDeviceProfileEvent {
 public:
  explicit OpDeviceProfileEvent(const platform::DeviceContext* device_context);
  virtual ~OpDeviceProfileEvent();

  // record event on host and device side (making a time stamp)
  void Record();

  // measure time lapse with respect to another event
  std::tuple<double, double> MeasureTimeLapseWrtOtherEvent(
      const OpDeviceProfileEvent& end_event) const;

 private:
  ///////////////////////////////////
  //        CUDA profiling         //
  ///////////////////////////////////
#if defined(PADDLE_WITH_CUDA)
  // (owned) CUDA event object
  cudaEvent_t event_obj_cuda_ = nullptr;
  // (not owned) which stream the event is recorded onto
  cudaStream_t cuda_stream_ = nullptr;
#endif

  ///////////////////////////////////
  //      HOST side profiling      //
  ///////////////////////////////////
  struct cpuEvent_t {
    // cpu can also be treated as a compute device
    double event_time_us_;
  };
  cpuEvent_t event_obj_cpu_;
};

class OpRuntimeProfiler {
  // Cross-platform event recorder for op runtime profiling.
  // For CPU device, recorder uses a simple timer to record op runtime.
  // For GPU device, recorder uses cuda event and stream to record op runtime.
 public:
  OpRuntimeProfiler();
  virtual ~OpRuntimeProfiler();

  // this will record event on both host and device side (if exist)
  void RecordEvent(const std::string& event_name,
                   const platform::DeviceContext* device_context);

  // return time lapse between two events in both host and device side
  std::tuple<double, double> MeasureTimeLapseBetweenEvents(
      const std::string& start_event_name,
      const std::string& end_event_name) const;

 protected:
  // mapping event name to event object, users just need to use event name
  // to record event of measure time lapse between events.
  std::unordered_map<std::string, std::shared_ptr<OpDeviceProfileEvent>>
      name_to_device_profile_events_;
};

}  // namespace profiler
}  // namespace framework
}  // namespace paddle
