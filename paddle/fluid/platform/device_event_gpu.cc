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
#include "paddle/fluid/platform/event.h"

#ifdef PADDLE_WITH_CUDA
namespace paddle {
namespace platform {
struct CUDADeviceEventWrapper {
  CUDADeviceEventWrapper(const platform::Place& place, unsigned int flag)
      : inner_event_(flag) {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(place), true,
        platform::errors::PreconditionNotMet(
            "Required device shall be CUDAPlace, but received %d. ", place));

    device_id_ = place.device;
    PADDLE_ENFORCE_GT(
        device_id_, -1,
        platform::errors::PreconditionNotMet(
            "Required DeviceOption.device_id > -1, but received %d. ",
            device_id_));
  }

  CudaEvent inner_event_;
  int device_id_;
};

void DeviceEventCreateCUDA(DeviceEvent* event, const platform::Place& place,
                           unsigned int flag) {
  event->InitEvent(std::make_shared<CUDADeviceEventWrapper>(place, flag));
}

void DeviceEventRecordCUDA(DeviceEvent* event, const DeviceContext* context) {
  auto* wrapper = static_cast<CUDADeviceEventWrapper*>(event->GetEvent().get());

  auto* cuda_dev_ctx =
      dynamic_cast<const platform::CUDADeviceContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      cuda_dev_ctx,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into CUDADeviceContext."));

  wrapper->inner_event_.Record(cuda_dev_ctx->stream());
}

bool DeviceEventQueryCUDA(const DeviceEvent* event) {
  auto* wrapper = static_cast<CUDADeviceEventWrapper*>(event->GetEvent().get());
  PADDLE_ENFORCE_NOT_NULL(
      wrapper,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast event into CUDADeviceEventWrapper."));

  return wrapper->inner_event_.Query();
}

void DeviceEventFinishCUDA(const DeviceEvent* event) {
  auto* wrapper = static_cast<CUDADeviceEventWrapper*>(event->GetEvent().get());
  // calling cudaEventSynchronize
  wrapper->inner_event_.Synchronize();
}

void DeviceEventCUDAWaitCUDA(const DeviceEvent* event,
                             const DeviceContext* context) {
  auto* wrapper = static_cast<CUDADeviceEventWrapper*>(event->GetEvent().get());
  auto* cuda_dev_ctx =
      dynamic_cast<const platform::CUDADeviceContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      cuda_dev_ctx,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into CUDADeviceContext."));
  // calling cudaStreamWaitEvent(stream, event, 0)
  cuda_dev_ctx->WaitEvent(wrapper->inner_event_.GetRawCudaEvent());
}

void DeviceEventCPUWaitCUDA(const DeviceEvent* event,
                            const DeviceContext* context) {
  DeviceEventFinishCUDA(event);
}

void DeviceEventSetFinishedCUDA(const DeviceEvent* event) {
  // do nothing
}

void EventResetCUDA(const DeviceEvent* event) {
  // do nothing
}

}  // namespace platform
}  // namespace paddle

using ::paddle::platform::kCUDA;
using ::paddle::platform::kCPU;
REGISTER_EVENT_CREATE_FUNCTION(kCUDA, paddle::platform::DeviceEventCreateCUDA)
REGISTER_EVENT_RECORD_FUNCTION(kCUDA, paddle::platform::DeviceEventRecordCUDA)
REGISTER_EVENT_QUERY_FUNCTION(kCUDA, paddle::platform::DeviceEventQueryCUDA)
REGISTER_EVENT_FINISH_FUNCTION(kCUDA, paddle::platform::DeviceEventFinishCUDA)
REGISTER_EVENT_SET_FINISHED_FUNCTION(
    kCUDA, paddle::platform::DeviceEventSetFinishedCUDA)
REGISTER_EVENT_WAIT_FUNCTION(kCUDA, kCUDA,
                             paddle::platform::DeviceEventCUDAWaitCUDA)
REGISTER_EVENT_WAIT_FUNCTION(kCPU, kCUDA,
                             paddle::platform::DeviceEventCPUWaitCUDA)
REGISTER_EVENT_RESET_FUNCTION(kCUDA, paddle::platform::EventResetCUDA)
#endif
