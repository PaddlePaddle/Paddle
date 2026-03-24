// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
#endif

#include <c10/core/DeviceType.h>

#include <string>
#include <utility>

#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device_event_base.h"

namespace c10 {
using DeviceIndex = int8_t;

struct Device final {
  using Type = DeviceType;
  Device() = default;
  Device(phi::Place place)
      : type_(PhiToDeviceType(place.GetType())),
        index_(place.GetType() == phi::AllocationType::CPU
                   ? static_cast<DeviceIndex>(-1)
                   : place.GetDeviceId()),
        custom_device_type_(place.GetDeviceType()) {}
  Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {}  // NOLINT
  Device(DeviceType type, DeviceIndex index, std::string custom_device_type)
      : type_(type),
        index_(index),
        custom_device_type_(std::move(custom_device_type)) {}  // NOLINT

  /// Constructs a `Device` from a string description, for convenience.
  /// The string supplied must follow the following schema:
  /// `(cpu|cuda)[:<device-index>]`
  /// where `cpu` or `cuda` specifies the device type, and
  /// `:<device-index>` optionally specifies a device index.
  /* implicit */ Device(const std::string& device_string);

  DeviceIndex index() const noexcept { return index_; }

  bool has_index() const noexcept { return index() != -1; }

  DeviceType type() const noexcept { return type_; }

  bool is_cuda() const noexcept { return type_ == DeviceType::CUDA; }

  bool is_cpu() const noexcept { return type_ == DeviceType::CPU; }

  std::string str() const;

  bool operator==(const Device& other) const noexcept {
    return type() == other.type() && this->index() == other.index() &&
           custom_device_type_ == other.custom_device_type_;
  }

  phi::Place _PD_GetInner() const {
    switch (type_) {
      case DeviceType::CPU:
        return phi::CPUPlace();
      case DeviceType::CUDA:
        return phi::GPUPlace(has_index() ? index_ : 0);
      case DeviceType::XPU:
        return phi::XPUPlace(has_index() ? index_ : 0);
      case DeviceType::IPU:
        return phi::IPUPlace(has_index() ? index_ : 0);
      case DeviceType::CUSTOM:
        return phi::CustomPlace(
            custom_device_type_.empty() ? "custom" : custom_device_type_,
            has_index() ? index_ : 0);
    }
    return phi::CPUPlace();
  }

 private:
  DeviceType type_{DeviceType::CPU};
  DeviceIndex index_{-1};
  std::string custom_device_type_;
};

std::ostream& operator<<(std::ostream& stream, const Device& device);

}  // namespace c10

namespace at {
using c10::Device;
using c10::DeviceIndex;
}  // namespace at

namespace torch {
using c10::Device;
using c10::DeviceIndex;
}  // namespace torch
