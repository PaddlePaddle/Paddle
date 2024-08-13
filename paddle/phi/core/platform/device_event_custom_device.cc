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

#ifdef PADDLE_WITH_CUSTOM_DEVICE

#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/core/platform/device/custom/custom_device_resource_pool.h"
#include "paddle/phi/core/platform/device_event_base.h"
namespace paddle {
namespace platform {
struct CustomDeviceEventWrapper {
  explicit CustomDeviceEventWrapper(const phi::Place& place) {
    PADDLE_ENFORCE_EQ(
        phi::is_custom_place(place),
        true,
        common::errors::PreconditionNotMet(
            "Required device shall be CustomPlace, but received %d. ", place));

    device_id_ = place.device;  // NOLINT
    PADDLE_ENFORCE_GT(
        device_id_,
        -1,
        common::errors::PreconditionNotMet(
            "Required DeviceOption.device_id > -1, but received %d. ",
            device_id_));
    inner_event_ =
        CustomDeviceEventResourcePool::Instance(place).New(device_id_);
  }
  std::shared_ptr<CustomDeviceEventObject> inner_event_;
  int device_id_;
};

void DeviceEventCreateCustomDevice(DeviceEvent* event,
                                   const phi::Place& place,
                                   unsigned int) {
  event->InitEvent(std::make_shared<CustomDeviceEventWrapper>(place));
}

void DeviceEventRecordCustomDevice(DeviceEvent* event,
                                   const DeviceContext* context) {
  auto* wrapper =
      static_cast<CustomDeviceEventWrapper*>(event->GetEvent().get());
  auto* custom_device_ctx = dynamic_cast<const phi::CustomContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      custom_device_ctx,
      common::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into NPUDeviceContext."));

  phi::stream::Stream stream_wrapper(custom_device_ctx->GetPlace(),
                                     custom_device_ctx->stream());
  wrapper->inner_event_->Record(&stream_wrapper);
}

bool DeviceEventQueryCustomDevice(const DeviceEvent* event) {
  auto* wrapper =
      static_cast<CustomDeviceEventWrapper*>(event->GetEvent().get());
  PADDLE_ENFORCE_NOT_NULL(
      wrapper,
      common::errors::PreconditionNotMet(
          "Failed to dynamic_cast event into CustomDeviceEventWrapper."));
  return wrapper->inner_event_->Query();
}

void DeviceEventFinishCustomDevice(const DeviceEvent* event) {
  auto* wrapper =
      static_cast<CustomDeviceEventWrapper*>(event->GetEvent().get());
  wrapper->inner_event_->Synchronize();
}

void DeviceEventCustomDeviceWaitCustomDevice(const DeviceEvent* event,
                                             const DeviceContext* context) {
  auto* wrapper =
      static_cast<CustomDeviceEventWrapper*>(event->GetEvent().get());
  auto* custom_device_ctx = dynamic_cast<const phi::CustomContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      custom_device_ctx,
      common::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into NPUDeviceContext."));
  phi::stream::Stream stream_wrapper(custom_device_ctx->GetPlace(),
                                     custom_device_ctx->stream());
  stream_wrapper.WaitEvent(wrapper->inner_event_.get());
}

void DeviceEventCPUWaitCustomDevice(const DeviceEvent* event,
                                    const DeviceContext* context) {
  DeviceEventFinishCustomDevice(event);
}

void DeviceEventSetFinishedCustomDevice(const DeviceEvent* event) {
  // do nothing
}

void EventResetCustomDevice(const DeviceEvent* event) {
  // do nothing
}

}  // namespace platform
}  // namespace paddle

using ::paddle::platform::kCPU;
using ::paddle::platform::kCUSTOM_DEVICE;
REGISTER_EVENT_CREATE_FUNCTION(kCUSTOM_DEVICE,
                               paddle::platform::DeviceEventCreateCustomDevice)
REGISTER_EVENT_RECORD_FUNCTION(kCUSTOM_DEVICE,
                               paddle::platform::DeviceEventRecordCustomDevice)
REGISTER_EVENT_QUERY_FUNCTION(kCUSTOM_DEVICE,
                              paddle::platform::DeviceEventQueryCustomDevice)
REGISTER_EVENT_FINISH_FUNCTION(kCUSTOM_DEVICE,
                               paddle::platform::DeviceEventFinishCustomDevice)
REGISTER_EVENT_SET_FINISHED_FUNCTION(
    kCUSTOM_DEVICE, paddle::platform::DeviceEventSetFinishedCustomDevice)
REGISTER_EVENT_WAIT_FUNCTION(
    kCUSTOM_DEVICE,
    kCUSTOM_DEVICE,
    paddle::platform::DeviceEventCustomDeviceWaitCustomDevice)
REGISTER_EVENT_WAIT_FUNCTION(kCPU,
                             kCUSTOM_DEVICE,
                             paddle::platform::DeviceEventCPUWaitCustomDevice)
REGISTER_EVENT_RESET_FUNCTION(kCUSTOM_DEVICE,
                              paddle::platform::EventResetCustomDevice)
#endif
