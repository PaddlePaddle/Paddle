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
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/event.h"

namespace paddle {
namespace platform {
#ifdef PADDLE_WITH_CUDA
struct CUDADeviceEventWrapper {
  explicit CUDADeviceEventWrapper(const DeviceOption& dev_opt) {
    PADDLE_ENFORCE_EQ(
        dev_opt.device_type(), static_cast<int>(DeviceType::kCUDA),
        platform::errors::PreconditionNotMet(
            "Required device type shall be CUDA, but received %d. ",
            dev_opt.device_type()));
    PADDLE_ENFORCE_GT(
        dev_opt.device_id(), -1,
        platform::errors::PreconditionNotMet(
            "Required DeviceOption.device_id > -1, but received %d. ",
            dev_opt.device_id()));
    device_id_ = dev_opt.device_id();
    inner_event_ = platform::CudaEvent();
  }

  CudaEvent inner_event_;
  int device_id_;
};

void DeviceEventCreateCUDA(DeviceEvent* event, const DeviceOption& dev_opt) {
  event->InitEvent(std::make_shared<CUDADeviceEventWrapper>(dev_opt));
}

void DeviceEventRecordCUDA(DeviceEvent* event, const platform::Place& place,
                           const void* context) {
  auto* wrapper = static_cast<CUDADeviceEventWrapper*>(event->GetEvent().get());
  auto* cuda_dev_ctx = static_cast<const platform::CUDADeviceContext*>(context);

  // TODO(Aurelius84): verify device_id and stream is as expected.
  wrapper->inner_event_.Record(cuda_dev_ctx->context()->Stream());
}

bool DeviceEventQueryCUDA(const DeviceEvent* event) {
  auto* wrapper = static_cast<CUDADeviceEventWrapper*>(event->GetEvent().get());
  return wrapper->inner_event_.Query();
}

REGISTER_EVENT_CREATE_FUNCTION(DeviceType::kCUDA, DeviceEventCreateCUDA)
REGISTER_EVENT_RECORD_FUNCTION(DeviceType::kCUDA, DeviceEventRecordCUDA)
REGISTER_EVENT_QUERY_FUNCTION(DeviceType::kCUDA, DeviceEventQueryCUDA)

#endif
}  // namespace platform
}  // namespace paddle
