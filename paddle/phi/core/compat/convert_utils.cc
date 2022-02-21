/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/compat/convert_utils.h"

#include "paddle/pten/backends/gpu/gpu_info.h"
#include "paddle/pten/backends/xpu/xpu_info.h"
#include "paddle/pten/common/place.h"
#include "paddle/pten/core/compat/op_utils.h"

namespace pten {

Backend TransToPtenBackend(const pten::Place& place) {
  if (place.GetType() == pten::AllocationType::CPU) {
    return Backend::CPU;
  } else if (place.GetType() == pten::AllocationType::GPU) {
    return Backend::GPU;
  } else {
    return Backend::UNDEFINED;
  }
}

pten::Place TransToPtenPlace(const Backend& backend, bool set_device_id) {
  // NOTE(zhiqiu): GetCurrentDeviceId not always success, and device id is not
  // always needed.
  // So, add set_device_id parameter here.
  switch (backend) {
    case pten::Backend::CPU:
      return pten::CPUPlace();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case pten::Backend::GPU:
      return pten::GPUPlace(
          set_device_id ? pten::backends::gpu::GetCurrentDeviceId() : 0);
#endif
#ifdef PADDLE_WITH_MKLDNN
    case pten::Backend::MKLDNN:
      return pten::CPUPlace();
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case pten::Backend::CUDNN:
      return pten::GPUPlace(
          set_device_id ? pten::backends::gpu::GetCurrentDeviceId() : 0);
#endif
#if defined(PADDLE_WITH_XPU)
    case pten::Backend::XPU:
      return pten::XPUPlace(
          set_device_id ? pten::backends::xpu::GetXPUCurrentDeviceId() : 0);
#endif
    default:
      PADDLE_THROW(pten::errors::Unimplemented(
          "Unsupported backend `%s` when casting it to paddle place type.",
          backend));
  }
}

std::string TransToPtenKernelName(const std::string& fluid_op_name) {
  return OpUtilsMap::Instance().GetBaseKernelName(fluid_op_name);
}

const std::string& TransToFluidOpName(const std::string& pten_kernel_name) {
  auto& base_kernel_name_map = OpUtilsMap::Instance().base_kernel_name_map();
  auto it = std::find_if(base_kernel_name_map.begin(),
                         base_kernel_name_map.end(),
                         [&pten_kernel_name](const auto& pair) {
                           return pair.second == pten_kernel_name;
                         });
  if (it != base_kernel_name_map.end()) {
    return it->first;
  }
  return pten_kernel_name;
}

}  // namespace pten
