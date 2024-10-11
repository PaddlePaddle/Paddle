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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/enforce.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif

COMMON_DECLARE_bool(pinned_memory_as_cpu_backend);

namespace phi {

Backend TransToPhiBackend(const phi::Place& place) {
  auto allocation_type = place.GetType();
  switch (allocation_type) {
    case phi::AllocationType::GPU:
      return Backend::GPU;
    case AllocationType::CPU:
      return Backend::CPU;
    case AllocationType::GPUPINNED: {
      if (FLAGS_pinned_memory_as_cpu_backend) {
        return Backend::CPU;
      } else {
        return Backend::GPU;
      }
    }
    case AllocationType::XPU:
      return Backend::XPU;
    case AllocationType::IPU:
      return Backend::IPU;
    case AllocationType::UNDEFINED:
      return Backend::UNDEFINED;
    case AllocationType::CUSTOM:
      return static_cast<Backend>(
          static_cast<size_t>(Backend::NUM_BACKENDS) +
          phi::CustomRegisteredDeviceMap::Instance()
              .GetOrRegisterGlobalDeviceTypeId(place.GetDeviceType()));
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported transform %s to phi Backend.", place));
  }
}

phi::Place TransToPhiPlace(const Backend& backend,
                           bool set_device_id,
                           const int8_t& device_id) {
  // NOTE(zhiqiu): GetCurrentDeviceId not always success, and device id is not
  // always needed.
  // So, add set_device_id parameter here.
  switch (backend) {
    case phi::Backend::CPU:
      return phi::CPUPlace();
    case phi::Backend::UNDEFINED:
      return phi::Place();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case phi::Backend::GPU:
    case phi::Backend::GPUDNN:
      return phi::GPUPlace(set_device_id
                               ? (device_id == -1
                                      ? phi::backends::gpu::GetCurrentDeviceId()
                                      : device_id)
                               : 0);
#endif
#ifdef PADDLE_WITH_DNNL
    case phi::Backend::ONEDNN:  // NOLINT
      return phi::CPUPlace();
#endif
#if defined(PADDLE_WITH_XPU)
    case phi::Backend::XPU:
      return phi::XPUPlace(
          set_device_id
              ? (device_id == -1 ? phi::backends::xpu::GetXPUCurrentDeviceId()
                                 : device_id)
              : 0);
#endif
    case phi::Backend::KPS:
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      return phi::GPUPlace(set_device_id
                               ? (device_id == -1
                                      ? phi::backends::gpu::GetCurrentDeviceId()
                                      : device_id)
                               : 0);
#elif defined(PADDLE_WITH_XPU_KP)
      return phi::XPUPlace(
          set_device_id
              ? (device_id == -1 ? phi::backends::xpu::GetXPUCurrentDeviceId()
                                 : device_id)
              : 0);
#endif
    default: {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      size_t device_type_id_ = static_cast<size_t>(backend) -
                               static_cast<size_t>(Backend::NUM_BACKENDS);
      if (backend == phi::Backend::CUSTOM) {
        device_type_id_ = 1;
      }
      std::string device_type =
          phi::CustomRegisteredDeviceMap::Instance().GetGlobalDeviceType(
              device_type_id_);
      if (!device_type.empty()) {
        return phi::CustomPlace(
            device_type,
            set_device_id ? phi::DeviceManager::GetDevice(device_type) : 0);
      } else if (backend == Backend::CUSTOM) {
        return phi::CustomPlace();
      }
#endif
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported backend `%s` when casting it to paddle place type.",
          backend));
    }
  }
}

const std::string& TransToPhiKernelName(const std::string& fluid_op_name) {
  return OpUtilsMap::Instance().GetBaseKernelName(fluid_op_name);
}

const std::string& TransToFluidOpName(const std::string& phi_kernel_name) {
  const auto& phi_kernel_to_fluid_op =
      OpUtilsMap::Instance().phi_kernel_to_fluid_op();
  auto it = phi_kernel_to_fluid_op.find(phi_kernel_name);
  if (it != phi_kernel_to_fluid_op.end()) {
    return it->second;
  }
  return phi_kernel_name;
}

#ifdef PADDLE_WITH_DNNL
dnnl::memory::data_type TransToOneDNNDataType(const phi::DataType& dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return dnnl::memory::data_type::f32;
    case DataType::BFLOAT16:
      return dnnl::memory::data_type::bf16;
    case DataType::INT8:
      return dnnl::memory::data_type::s8;
    case DataType::UINT8:
      return dnnl::memory::data_type::u8;
    case DataType::INT32:
      return dnnl::memory::data_type::s32;
    default:
      return dnnl::memory::data_type::undef;
  }
}
#endif

}  // namespace phi
