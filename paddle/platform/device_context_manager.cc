/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/platform/device_context_manager.h"

namespace paddle {
namespace platform {

DeviceContextManager::DeviceContextManager() {
#ifndef PADDLE_ONLY_CPU
  device_count_ = GetDeviceCount();
  cuda_contexts_.reserve(device_count_);
  for (int i = 0; i < device_count_; i++) {
    cuda_contexts_[i] = nullptr;
  }
#endif
}

template <>
CPUDeviceContext* DeviceContextManager::GetDeviceContext<
    CPUPlace, CPUDeviceContext>(const CPUPlace& place) {
  if (!cpu_context_) {
    cpu_context_.reset(new CPUDeviceContext(place));
  }
  return cpu_context_.get();
}

#ifndef PADDLE_ONLY_CPU
template <>
CUDADeviceContext* DeviceContextManager::GetDeviceContext<
    GPUPlace, CUDADeviceContext>(const GPUPlace& place) {
  int gpu_id = place.device;
  PADDLE_ENFORCE(gpu_id < device_count_,
                 "GPU device id must less than device count");
  SetDeviceId(gpu_id);
  auto ctx = cuda_contexts_[gpu_id];
  if (!ctx) {
    ctx.reset(new CUDADeviceContext(gpu_place));
  }
  return ctx.get();
}
#endif

}  // namespace platform
}  // namespace paddle
