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

#include "paddle/platform/device_context.h"

namespace paddle {
namespace platform {

template <>
Eigen::DefaultDevice* DeviceContext::get_eigen_device<Eigen::DefaultDevice>()
    const {
  return reinterpret_cast<const CPUDeviceContext*>(this)->eigen_device();
}

#ifndef PADDLE_ONLY_CPU
template <>
Eigen::GpuDevice* DeviceContext::get_eigen_device<Eigen::GpuDevice>() const {
  return reinterpret_cast<const CUDADeviceContext*>(this)->eigen_device();
}
#endif

}  // namespace platform
}  // namespace paddle
