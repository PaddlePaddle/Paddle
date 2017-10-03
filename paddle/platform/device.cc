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

#include "paddle/platform/device.h"

namespace paddle {
namespace platform {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

CPUDeviceContext* GetCPUDeviceContext(const CPUPlace& place) {
  static std::unique_ptr<CPUDeviceContext> g_cpu_device_context =
      make_unique<CPUDeviceContext>(place);
  return g_cpu_device_context.get();
}

#ifndef PADDLE_ONLY_CPU
CUDADeviceContext* GetCUDADeviceContext(const GPUPlace& place) {
  static std::unique_ptr<CUDADeviceContext> g_cuda_device_context =
      make_unique<CUDADeviceContext>(place);
  return g_cuda_device_context.get();
}
#endif

Device* GetDevice(const Place& place) {
  CPUPlace cpu_place;
#ifndef PADDLE_ONLY_CPU
  if (is_gpu_place(place)) {
    GPUPlace gpu_place = boost::get<GPUPlace>(place);
    static std::unique_ptr<Device> g_device = make_unique<Device>(
        GetCPUDeviceContext(cpu_place), GetCUDADeviceContext(gpu_place));
    return g_device.get();
  } else {
    static std::unique_ptr<Device> g_device =
        make_unique<Device>(GetCPUDeviceContext(cpu_place), nullptr);
    return g_device.get();
  }
#else
  static std::unique_ptr<Device> g_device =
      make_unique<Device>(GetCPUDeviceContext(cpu_place));
  return g_device.get();
#endif
}
}  // namespace platform
}  // namespace paddle
