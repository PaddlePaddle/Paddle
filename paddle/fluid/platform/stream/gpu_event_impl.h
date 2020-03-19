//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cuda_runtime.h>
#include "paddle/fluid/platform/stream/gpu_event.h"
#include "paddle/fluid/platform/stream/stream_internal.h"

namespace paddle {
namespace platform {
namespace stream {

class GpuStream;

class GpuEvent : public internal::EventInterface {
 public:
  GpuEvent();
  virtual ~GpuEvent();
  bool Init();
  bool InsertEvent(GpuStream* stream);
  Event::Status GetEventStatus();
  cudaEvent_t gpu_event();
  void Destroy();

 private:
  cudaEvent_t gpu_event_;
};

}  // namespace stream
}  // namespace platform
}  // namespace paddle
