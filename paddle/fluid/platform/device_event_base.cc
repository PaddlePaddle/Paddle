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

#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/fluid/platform/device_event_cpu.h"
#include "paddle/fluid/platform/event.h"

namespace paddle {
namespace platform {

EventCreateFunction DeviceEvent::event_creator_[MaxDeviceTypes];
EventRecordFunction DeviceEvent::event_recorder_[MaxDeviceTypes];
EventQueryFunction DeviceEvent::event_querier_[MaxDeviceTypes];
EventFinishFunction DeviceEvent::event_finisher_[MaxDeviceTypes];
EventFinishFunction DeviceEvent::event_finished_setter_[MaxDeviceTypes];
EventWaitFunction DeviceEvent::event_waiter_[MaxDeviceTypes][MaxDeviceTypes];

/*
 * Generate flag used to create event on all sorts of equipment.
 * NOTE: Support CPU/CUDA/ROCM currently.
 */
unsigned int GenerateDeviceEventFlag(bool enable_timing, bool blocking,
                                     bool interprocess) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return get_cuda_flags(enable_timing, blocking, interprocess);
#endif
  return 0;
}

void DeviceEventCreateCPU(DeviceEvent* event, const platform::Place& place,
                          unsigned int flag) {
  event->InitEvent(std::make_shared<CPUDeviceEventWrapper>(place, flag));
}

void DeviceEventRecordCPU(DeviceEvent* event, const DeviceContext* context) {
  auto* wrapper = static_cast<CPUDeviceEventWrapper*>(event->GetEvent().get());

  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  PADDLE_ENFORCE_NE(wrapper->status_.load(), EventStatus::SCHEDULED,
                    platform::errors::PreconditionNotMet(
                        "EventStatus shall be not SCHEDULED before Record()"));
  if (wrapper->status_ == EventStatus::INITIALIZED) {
    wrapper->status_ = EventStatus::SCHEDULED;
  }
}

bool DeviceEventQueryCPU(const DeviceEvent* event) {
  auto* wrapper = static_cast<CPUDeviceEventWrapper*>(event->GetEvent().get());
  PADDLE_ENFORCE_NOT_NULL(
      wrapper, platform::errors::PreconditionNotMet(
                   "Failed to dynamic_cast event into CPUDeviceEventWrapper."));

  return wrapper->status_ == EventStatus::SUCCESS;
}

void DeviceEventFinishCPU(const DeviceEvent* event) {
  auto* wrapper = static_cast<CPUDeviceEventWrapper*>(event->GetEvent().get());

  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  while (wrapper->status_ != EventStatus::SUCCESS &&
         wrapper->status_ != EventStatus::FAILED) {
    wrapper->cv_completed_.wait(lock);
  }
}

void DeviceEventCPUWaitCPU(const DeviceEvent* event,
                           const DeviceContext* context) {
  DeviceEventFinishCPU(event);
}

void EventSetFinishedCPU(const DeviceEvent* event) {
  auto* wrapper = static_cast<CPUDeviceEventWrapper*>(event->GetEvent().get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);

  PADDLE_ENFORCE_LE(wrapper->status_.load(), EventStatus::SCHEDULED,
                    platform::errors::PreconditionNotMet(
                        "EventStatus shall be  INITIALIZED | SCHEDULED before "
                        "EventSetFinishedCPU()"));
  wrapper->status_ = EventStatus::SUCCESS;
  wrapper->cv_completed_.notify_all();
}

}  // namespace platform
}  // namespace paddle

using ::paddle::platform::kCPU;
REGISTER_EVENT_CREATE_FUNCTION(kCPU, paddle::platform::DeviceEventCreateCPU)
REGISTER_EVENT_RECORD_FUNCTION(kCPU, paddle::platform::DeviceEventRecordCPU)
REGISTER_EVENT_QUERY_FUNCTION(kCPU, paddle::platform::DeviceEventQueryCPU)
REGISTER_EVENT_FINISH_FUNCTION(kCPU, paddle::platform::DeviceEventFinishCPU)
REGISTER_EVENT_SET_FINISHED_FUNCTION(kCPU,
                                     paddle::platform::EventSetFinishedCPU);
REGISTER_EVENT_WAIT_FUNCTION(kCPU, kCPU,
                             paddle::platform::DeviceEventCPUWaitCPU)
