/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/mlu/mlu_stream.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace paddle {
namespace platform {
namespace stream {

bool MLUStream::Init(const MLUPlace& place, const int priority) {
  PADDLE_ENFORCE_EQ(is_mlu_place(place), true,
                    platform::errors::InvalidArgument(
                        "MLU stream must be created using mlu place."));
  place_ = place;
  MLUDeviceGuard guard(place_.device);
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueCreate(&stream_));
  callback_manager_.reset(new StreamCallbackManager<mluStream>(stream_));
  VLOG(3) << "MLUStream Init stream: " << stream_;
  return true;
}

void MLUStream::Destroy() {
  MLUDeviceGuard guard(place_.device);
  Wait();
  WaitCallback();
  if (stream_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueDestroy(stream_));
  }
  stream_ = nullptr;
}

void MLUStream::Wait() const {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueSync(stream_));
}

MLUStream* get_current_mlu_stream(int deviceId) {
#ifdef PADDLE_WITH_MLU
  if (deviceId == -1) {
    deviceId = platform::GetMLUCurrentDeviceId();
  }
  auto& pool = platform::DeviceContextPool::Instance();
  platform::Place device = MLUPlace(deviceId);
  auto stream = static_cast<platform::MLUDeviceContext*>(pool.Get(device))
                    ->context()
                    ->Stream()
                    .get();
  return stream;
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "Paddle is not compiled with MLU. Cannot visit mlu current stream."));
  return nullptr;
#endif
}

MLUStream* set_current_mlu_stream(MLUStream* stream) {
#ifdef PADDLE_WITH_MLU
  auto& device = stream->GetPlace();
  auto& pool = platform::DeviceContextPool::Instance();
  return static_cast<platform::MLUDeviceContext*>(pool.Get(device))
      ->context()
      ->SetStream(stream);
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "Paddle is not compiled with MLU. Cannot visit mlu current stream."));
  return nullptr;
#endif
}
}  // namespace stream
}  // namespace platform
}  // namespace paddle
