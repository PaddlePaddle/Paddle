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

#ifdef PADDLE_WITH_ASCEND_CL

#include "paddle/fluid/platform/device/npu/npu_resource_pool.h"
#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/fluid/platform/event.h"
namespace paddle {
namespace platform {
struct NPUDeviceEventWrapper {
  explicit NPUDeviceEventWrapper(const platform::Place& place) {
    PADDLE_ENFORCE_EQ(
        platform::is_npu_place(place),
        true,
        platform::errors::PreconditionNotMet(
            "Required device shall be NPUPlace, but received %d. ", place));

    device_id_ = place.device;
    PADDLE_ENFORCE_GT(
        device_id_,
        -1,
        platform::errors::PreconditionNotMet(
            "Required DeviceOption.device_id > -1, but received %d. ",
            device_id_));
    inner_event_ = NpuEventResourcePool::Instance().New(device_id_);
  }
  std::shared_ptr<NpuEventObject> inner_event_;
  int device_id_;
};

void DeviceEventCreateNPU(DeviceEvent* event,
                          const platform::Place& place,
                          unsigned int) {
  event->InitEvent(std::make_shared<NPUDeviceEventWrapper>(place));
}

void DeviceEventRecordNPU(DeviceEvent* event, const DeviceContext* context) {
  auto* wrapper = static_cast<NPUDeviceEventWrapper*>(event->GetEvent().get());
  auto* npu_dev_ctx = dynamic_cast<const platform::NPUDeviceContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      npu_dev_ctx,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into NPUDeviceContext."));
  NPUEventRecord(wrapper->inner_event_.get(), npu_dev_ctx->stream());
}

bool DeviceEventQueryNPU(const DeviceEvent* event) {
  auto* wrapper = static_cast<NPUDeviceEventWrapper*>(event->GetEvent().get());
  PADDLE_ENFORCE_NOT_NULL(
      wrapper,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast event into NPUDeviceEventWrapper."));
  aclrtEventStatus status = ACL_EVENT_STATUS_COMPLETE;
  platform::NPUEventQuery(wrapper->inner_event_.get(), &status);
  return ACL_EVENT_STATUS_COMPLETE == status;
}

void DeviceEventFinishNPU(const DeviceEvent* event) {
  auto* wrapper = static_cast<NPUDeviceEventWrapper*>(event->GetEvent().get());
  NPUEventSynchronize(wrapper->inner_event_.get());
}

void DeviceEventNPUWaitNPU(const DeviceEvent* event,
                           const DeviceContext* context) {
  auto* wrapper = static_cast<NPUDeviceEventWrapper*>(event->GetEvent().get());
  auto* npu_dev_ctx = dynamic_cast<const platform::NPUDeviceContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      npu_dev_ctx,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into NPUDeviceContext."));
  NPUStreamWaitEvent(npu_dev_ctx->stream(), wrapper->inner_event_.get());
}

void DeviceEventCPUWaitNPU(const DeviceEvent* event,
                           const DeviceContext* context) {
  DeviceEventFinishNPU(event);
}

void DeviceEventSetFinishedNPU(const DeviceEvent* event) {
  // do nothing
}

void EventResetNPU(const DeviceEvent* event) {
  // do nothing
}

}  // namespace platform
}  // namespace paddle

using ::paddle::platform::kCPU;
using ::paddle::platform::kNPU;
REGISTER_EVENT_CREATE_FUNCTION(kNPU, paddle::platform::DeviceEventCreateNPU)
REGISTER_EVENT_RECORD_FUNCTION(kNPU, paddle::platform::DeviceEventRecordNPU)
REGISTER_EVENT_QUERY_FUNCTION(kNPU, paddle::platform::DeviceEventQueryNPU)
REGISTER_EVENT_FINISH_FUNCTION(kNPU, paddle::platform::DeviceEventFinishNPU)
REGISTER_EVENT_SET_FINISHED_FUNCTION(
    kNPU, paddle::platform::DeviceEventSetFinishedNPU)
REGISTER_EVENT_WAIT_FUNCTION(kNPU,
                             kNPU,
                             paddle::platform::DeviceEventNPUWaitNPU)
REGISTER_EVENT_WAIT_FUNCTION(kCPU,
                             kNPU,
                             paddle::platform::DeviceEventCPUWaitNPU)
REGISTER_EVENT_RESET_FUNCTION(kNPU, paddle::platform::EventResetNPU)
#endif
