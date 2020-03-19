//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/stream/gpu_event_impl.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/stream/gpu_stream.h"

namespace paddle {
namespace platform {
namespace stream {

bool GpuEvent::Init() {
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventCreate(&gpu_event_));
  return true;
}

bool GpuEvent::InsertEvent(GpuStream* stream) {
  VLOG(3) << "stream:" << stream << " insert cuda event:" << gpu_event_;
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaEventRecord(gpu_event_, stream->gpu_stream()));
  return true;
}

cudaEvent_t GpuEvent::gpu_event() { return gpu_event_; }

GpuEvent::GpuEvent() : gpu_event_(nullptr) {}

GpuEvent::~GpuEvent() {}

Event::Status GpuEvent::GetEventStatus() {
  auto res = cudaEventQuery(gpu_event_);
  VLOG(3) << "Get cuda event:" << gpu_event_ << " status:" << res;
  switch (res) {
    case cudaSuccess:
      return Event::Status::kComplete;
    case cudaErrorNotReady:
      return Event::Status::kPending;
    default:
      LOG(INFO) << "Error returned for event status: " << res;
      return Event::Status::kError;
  }
}

void GpuEvent::Destroy() {
  if (gpu_event_) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventDestroy(gpu_event_));
    gpu_event_ = nullptr;
  }
}

}  // namespace stream
}  // namespace platform
}  // namespace paddle
