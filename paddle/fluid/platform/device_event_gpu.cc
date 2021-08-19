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

#include "paddle/fluid/platform/device_event.h"
#include "paddle/fluid/platform/event.h"

#ifdef PADDLE_WITH_CUDA
namespace paddle {
namespace platform {
struct CUDADeviceEventWrapper {
  explicit CUDADeviceEventWrapper(const DeviceOption& dev_opt)
      : inner_event_() {
    PADDLE_ENFORCE_EQ(
        dev_opt.device_type(), static_cast<int>(DeviceType::CUDA),
        platform::errors::PreconditionNotMet(
            "Required device type shall be CUDA, but received %d. ",
            dev_opt.device_type()));
    PADDLE_ENFORCE_GT(
        dev_opt.device_id(), -1,
        platform::errors::PreconditionNotMet(
            "Required DeviceOption.device_id > -1, but received %d. ",
            dev_opt.device_id()));
    device_id_ = dev_opt.device_id();
  }

  CudaEvent inner_event_;
  int device_id_;
};

void DeviceEventCreateCUDA(DeviceEvent* event, const DeviceOption& dev_opt) {
  event->InitEvent(std::make_shared<CUDADeviceEventWrapper>(dev_opt));
}

void DeviceEventRecordCUDA(DeviceEvent* event, const platform::Place& place,
                           const DeviceContext* context) {
  auto* wrapper = static_cast<CUDADeviceEventWrapper*>(event->GetEvent().get());

  auto* cuda_dev_ctx =
      dynamic_cast<const platform::CUDADeviceContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      cuda_dev_ctx,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into CUDADeviceContext."));

  wrapper->inner_event_.Record(*cuda_dev_ctx->context()->Stream());
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

void DeviceEventCUDAWaitCUDA(const DeviceEvent* event, DeviceContext* context) {
  auto* wrapper = static_cast<CUDADeviceEventWrapper*>(event->GetEvent().get());
  auto* cuda_dev_ctx =
      dynamic_cast<const platform::CUDADeviceContext*>(context);
  PADDLE_ENFORCE_NOT_NULL(
      cuda_dev_ctx,
      platform::errors::PreconditionNotMet(
          "Failed to dynamic_cast context into CUDADeviceContext."));
  // calling cudaStreamWaitEvent(stream, event, 0)
  cuda_dev_ctx->context()->Stream()->WaitEvent(
      wrapper->inner_event_.GetRawCudaEvent());
}

void DeviceEventCPUWaitCUDA(const DeviceEvent* event, DeviceContext* context) {
  DeviceEventFinishCUDA(event);
}

}  // namespace platform
}  // namespace paddle

using ::paddle::platform::kCUDA;
using ::paddle::platform::kCPU;
REGISTER_EVENT_CREATE_FUNCTION(kCUDA, paddle::platform::DeviceEventCreateCUDA)
REGISTER_EVENT_RECORD_FUNCTION(kCUDA, paddle::platform::DeviceEventRecordCUDA)
REGISTER_EVENT_QUERY_FUNCTION(kCUDA, paddle::platform::DeviceEventQueryCUDA)
REGISTER_EVENT_FINISH_FUNCTION(kCUDA, paddle::platform::DeviceEventFinishCUDA)
REGISTER_EVENT_WAIT_FUNCTION(kCUDA, kCUDA,
                             paddle::platform::DeviceEventCUDAWaitCUDA)
REGISTER_EVENT_WAIT_FUNCTION(kCPU, kCUDA,
                             paddle::platform::DeviceEventCPUWaitCUDA)
#endif
