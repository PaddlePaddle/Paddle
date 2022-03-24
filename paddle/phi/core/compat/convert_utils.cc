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

#include "paddle/phi/core/compat/convert_utils.h"

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/op_utils.h"

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif

namespace phi {

Backend TransToPhiBackend(const phi::Place& place) {
  auto allocation_type = place.GetType();
  if (allocation_type == phi::AllocationType::CPU) {
    return Backend::CPU;
  } else if (allocation_type == phi::AllocationType::GPU) {
    return Backend::GPU;
  } else if (allocation_type == phi::AllocationType::XPU) {
    return Backend::XPU;
  } else if (allocation_type == phi::AllocationType::CUSTOM) {
    return static_cast<Backend>(
        static_cast<size_t>(Backend::NUM_BACKENDS) +
        GetOrRegisterGlobalDeviceTypeId(place.GetDeviceType()));
  } else {
    return Backend::UNDEFINED;
  }
}

phi::Place TransToPhiPlace(const Backend& backend, bool set_device_id) {
  // NOTE(zhiqiu): GetCurrentDeviceId not always success, and device id is not
  // always needed.
  // So, add set_device_id parameter here.
  switch (backend) {
    case phi::Backend::CPU:
      return phi::CPUPlace();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case phi::Backend::GPU:
      return phi::GPUPlace(
          set_device_id ? phi::backends::gpu::GetCurrentDeviceId() : 0);
#endif
#ifdef PADDLE_WITH_MKLDNN
    case phi::Backend::MKLDNN:
      return phi::CPUPlace();
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case phi::Backend::GPUDNN:
      return phi::GPUPlace(
          set_device_id ? phi::backends::gpu::GetCurrentDeviceId() : 0);
#endif
#if defined(PADDLE_WITH_XPU)
    case phi::Backend::XPU:
      return phi::XPUPlace(
          set_device_id ? phi::backends::xpu::GetXPUCurrentDeviceId() : 0);
#endif
    case phi::Backend::KPS:
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      return phi::GPUPlace(
          set_device_id ? phi::backends::gpu::GetCurrentDeviceId() : 0);
#elif defined(PADDLE_WITH_XPU_KP)
      return phi::XPUPlace(
          set_device_id ? phi::backends::xpu::GetXPUCurrentDeviceId() : 0);
#endif
    default: {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      size_t device_type_id_ = static_cast<size_t>(backend) -
                               static_cast<size_t>(Backend::NUM_BACKENDS);
      std::string device_type = phi::GetGlobalDeviceType(device_type_id_);
      if (!device_type.empty()) {
        return phi::CustomPlace(
            device_type,
            set_device_id ? phi::DeviceManager::GetDevice(device_type) : 0);
      }
#endif
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported backend `%s` when casting it to paddle place type.",
          backend));
    }
  }
}

std::string TransToPhiKernelName(const std::string& fluid_op_name) {
  return OpUtilsMap::Instance().GetBaseKernelName(fluid_op_name);
}

const std::string& TransToFluidOpName(const std::string& phi_kernel_name) {
  auto& base_kernel_name_map = OpUtilsMap::Instance().base_kernel_name_map();
  auto it = std::find_if(base_kernel_name_map.begin(),
                         base_kernel_name_map.end(),
                         [&phi_kernel_name](const auto& pair) {
                           return pair.second == phi_kernel_name;
                         });
  if (it != base_kernel_name_map.end()) {
    return it->first;
  }
  return phi_kernel_name;
}

}  // namespace phi
