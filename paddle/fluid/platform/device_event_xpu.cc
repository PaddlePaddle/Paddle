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

#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device_event_base.h"

#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace platform {

struct XPUDeviceEventWrapper {
  explicit XPUDeviceEventWrapper(const platform::Place& place) {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(place),
        true,
        platform::errors::PreconditionNotMet(
            "Required device shall be XPUPlace, but received %d. ", place));

    device_id_ = place.device;
    PADDLE_ENFORCE_GT(
        device_id_,
        -1,
        platform::errors::PreconditionNotMet(
            "Required DeviceOption.device_id > -1, but received %d. ",
            device_id_));
    xpu_event_create(&handle_);
  }

  xpuEventHandle handle_;
  int device_id_;
};

void DeviceEventCreateXPU(DeviceEvent* event,
                          const platform::Place& place,
                          unsigned int) {
  event->InitEvent(std::make_shared<XPUDeviceEventWrapper>(place));
}

void DeviceEventRecordXPU(DeviceEvent* event, const DeviceContext* context) {
  auto* wrapper = static_cast<XPUDeviceEventWrapper*>(event->GetEvent().get());
  PADDLE_ENFORCE_NOT_NULL(
      wrapper,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast event into XPUDeviceEventWrapper."));

  auto* xpu_dev_ctx = dynamic_cast<const platform::XPUDeviceContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      xpu_dev_ctx,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into XPUDeviceContext."));
  xpu_event_record(wrapper->handle_, xpu_dev_ctx->stream());
}

void DeviceEventFinishXPU(const DeviceEvent* event) {
  auto* wrapper = static_cast<XPUDeviceEventWrapper*>(event->GetEvent().get());
  PADDLE_ENFORCE_NOT_NULL(
      wrapper,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast event into XPUDeviceEventWrapper."));
  xpu_event_wait(wrapper->handle_);
}

// current xpu not support query, used wait to instead.
bool DeviceEventQueryXPU(const DeviceEvent* event) {
  DeviceEventFinishXPU(event);
  return true;
}

void DeviceEventXPUWaitXPU(const DeviceEvent* event,
                           const DeviceContext* context) {
  auto* wrapper = static_cast<XPUDeviceEventWrapper*>(event->GetEvent().get());
  PADDLE_ENFORCE_NOT_NULL(
      wrapper,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast event into XPUDeviceEventWrapper."));
  auto* xpu_dev_ctx = dynamic_cast<const platform::XPUDeviceContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      xpu_dev_ctx,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into XOUDeviceContext."));
  xpu_stream_wait_event(xpu_dev_ctx->stream(), wrapper->handle_);
}

void DeviceEventCPUWaitXPU(const DeviceEvent* event,
                           const DeviceContext* context) {
  DeviceEventFinishXPU(event);
}

void DeviceEventSetFinishedXPU(const DeviceEvent* event) {
  // do nothing
}

void EventResetXPU(const DeviceEvent* event) {
  // do nothing
}

}  // namespace platform
}  // namespace paddle

using ::paddle::platform::kCPU;
using ::paddle::platform::kXPU;
REGISTER_EVENT_CREATE_FUNCTION(kXPU, paddle::platform::DeviceEventCreateXPU)
REGISTER_EVENT_RECORD_FUNCTION(kXPU, paddle::platform::DeviceEventRecordXPU)
REGISTER_EVENT_QUERY_FUNCTION(kXPU, paddle::platform::DeviceEventQueryXPU)
REGISTER_EVENT_FINISH_FUNCTION(kXPU, paddle::platform::DeviceEventFinishXPU)
REGISTER_EVENT_SET_FINISHED_FUNCTION(
    kXPU, paddle::platform::DeviceEventSetFinishedXPU)
REGISTER_EVENT_WAIT_FUNCTION(kXPU,
                             kXPU,
                             paddle::platform::DeviceEventXPUWaitXPU)
REGISTER_EVENT_WAIT_FUNCTION(kCPU,
                             kXPU,
                             paddle::platform::DeviceEventCPUWaitXPU)
REGISTER_EVENT_RESET_FUNCTION(kXPU, paddle::platform::EventResetXPU)
#endif
