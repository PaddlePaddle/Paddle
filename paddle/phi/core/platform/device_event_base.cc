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

#include "paddle/phi/core/platform/device_event_base.h"

#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/core/platform/device_event_cpu.h"

namespace paddle {
namespace platform {

EventCreateFunction DeviceEvent::event_creator_[MaxDeviceTypes];   // NOLINT
EventRecordFunction DeviceEvent::event_recorder_[MaxDeviceTypes];  // NOLINT
EventQueryFunction DeviceEvent::event_querier_[MaxDeviceTypes];    // NOLINT
EventFinishFunction DeviceEvent::event_finisher_[MaxDeviceTypes];  // NOLINT
EventSetFinishedFunction                                           // NOLINT
    DeviceEvent::event_finished_setter_[MaxDeviceTypes];
EventWaitFunction DeviceEvent::event_waiter_[MaxDeviceTypes]  // NOLINT
                                            [MaxDeviceTypes];
EventResetFunction DeviceEvent::event_resetter_[MaxDeviceTypes];  // NOLINT

/*
 * Generate flag used to create event on all sorts of equipment.
 * NOTE: Support CPU/CUDA/ROCM currently.
 */
unsigned int GenerateDeviceEventFlag(bool enable_timing,
                                     bool blocking,
                                     bool interprocess) {
#ifdef PADDLE_WITH_CUDA
  unsigned int flags =
      (blocking ? cudaEventBlockingSync : cudaEventDefault) |
      (enable_timing ? cudaEventDefault : cudaEventDisableTiming) |
      (interprocess ? cudaEventInterprocess : cudaEventDefault);
  return flags;
#endif

#ifdef PADDLE_WITH_HIP
  unsigned int flags =
      (blocking ? hipEventBlockingSync : hipEventDefault) |
      (enable_timing ? hipEventDefault : hipEventDisableTiming) |
      (interprocess ? hipEventInterprocess : hipEventDefault);
  return flags;
#endif

  return 0;
}

void DeviceEventCreateCPU(DeviceEvent* event,
                          const phi::Place& place,
                          unsigned int flag) {
  event->InitEvent(std::make_shared<CPUDeviceEventWrapper>(place, flag));
}

void DeviceEventRecordCPU(DeviceEvent* event, const DeviceContext* context) {
  auto* wrapper = static_cast<CPUDeviceEventWrapper*>(event->GetEvent().get());

  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  // NOTE: As for CudaEvent_t, it can be used to Record() repeatedly.
  // CudaEvent_t internally reset its status from finished into initialized. So
  // we simulate the process here.
  if (wrapper->status_.load() == EventStatus::SUCCESS) {
    VLOG(3) << "Found EventStatus is SUCCESS before RecordCPU. Reset it into "
               "INITIALIZED.";
    wrapper->status_ = EventStatus::INITIALIZED;
  }

  PADDLE_ENFORCE_LT(
      wrapper->status_.load(),
      EventStatus::SCHEDULED,
      common::errors::PreconditionNotMet(
          "EventStatus shall be not SCHEDULED before Record(), but received %s",
          wrapper->status_.load()));
  if (wrapper->status_ == EventStatus::INITIALIZED) {
    wrapper->status_ = EventStatus::SCHEDULED;
  }
}

bool DeviceEventQueryCPU(const DeviceEvent* event) {
  auto* wrapper = static_cast<CPUDeviceEventWrapper*>(event->GetEvent().get());
  PADDLE_ENFORCE_NOT_NULL(
      wrapper,
      common::errors::PreconditionNotMet(
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

  PADDLE_ENFORCE_LE(wrapper->status_.load(),
                    EventStatus::SCHEDULED,
                    common::errors::PreconditionNotMet(
                        "EventStatus shall be  INITIALIZED | SCHEDULED before "
                        "EventSetFinishedCPU()"));
  wrapper->status_ = EventStatus::SUCCESS;
  wrapper->cv_completed_.notify_all();
}

void EventResetCPU(const DeviceEvent* event) {
  auto* wrapper = static_cast<CPUDeviceEventWrapper*>(event->GetEvent().get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  wrapper->status_ = EventStatus::INITIALIZED;
}

}  // namespace platform
}  // namespace paddle

using ::paddle::platform::kCPU;
REGISTER_EVENT_CREATE_FUNCTION(kCPU, paddle::platform::DeviceEventCreateCPU)
REGISTER_EVENT_RECORD_FUNCTION(kCPU, paddle::platform::DeviceEventRecordCPU)
REGISTER_EVENT_QUERY_FUNCTION(kCPU, paddle::platform::DeviceEventQueryCPU)
REGISTER_EVENT_FINISH_FUNCTION(kCPU, paddle::platform::DeviceEventFinishCPU)
REGISTER_EVENT_SET_FINISHED_FUNCTION(kCPU,
                                     paddle::platform::EventSetFinishedCPU)
REGISTER_EVENT_WAIT_FUNCTION(kCPU,
                             kCPU,
                             paddle::platform::DeviceEventCPUWaitCPU)
REGISTER_EVENT_RESET_FUNCTION(kCPU, paddle::platform::EventResetCPU)
