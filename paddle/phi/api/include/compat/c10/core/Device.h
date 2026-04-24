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
#include <c10/util/Exception.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <utility>

#include "paddle/common/macros.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device_event_base.h"

namespace c10 {
using DeviceIndex = int8_t;

struct PADDLE_API Device final {
  using Type = DeviceType;
  Device() = default;
  Device(phi::Place place)
      : type_(PhiToDeviceType(place.GetType())),
        index_(place.GetType() == phi::AllocationType::CPU
                   ? static_cast<DeviceIndex>(-1)
                   : place.GetDeviceId()),
        custom_device_type_(place.GetDeviceType()) {
    validate();
  }
  Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {  // NOLINT
    validate();
  }
  Device(DeviceType type, DeviceIndex index, std::string custom_device_type)
      : type_(type),
        index_(index),
        custom_device_type_(std::move(custom_device_type)) {  // NOLINT
    validate();
  }

  /// Constructs a `Device` from a string description, for convenience.
  /// The string supplied must follow the following schema:
  /// `(cpu|cuda)[:<device-index>]`
  /// where `cpu` or `cuda` specifies the device type, and
  /// `:<device-index>` optionally specifies a device index.
  /* implicit */ Device(const std::string& device_string);

  DeviceIndex index() const noexcept { return index_; }

  bool has_index() const noexcept { return index() != -1; }

  DeviceType type() const noexcept { return type_; }

  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

  void set_index(DeviceIndex index) {
    index_ = index;
    validate();
  }

  bool is_cuda() const noexcept { return type_ == DeviceType::CUDA; }

  bool is_privateuseone() const noexcept {
    return type_ == DeviceType::PrivateUse1;
  }

  bool is_mps() const noexcept { return false; }

  bool is_hip() const noexcept { return false; }

  bool is_ve() const noexcept { return false; }

  bool is_xpu() const noexcept { return type_ == DeviceType::XPU; }

  bool is_ipu() const noexcept { return type_ == DeviceType::IPU; }

  bool is_xla() const noexcept { return false; }

  bool is_mtia() const noexcept { return false; }

  bool is_hpu() const noexcept { return false; }

  bool is_lazy() const noexcept { return false; }

  bool is_vulkan() const noexcept { return false; }

  bool is_metal() const noexcept { return false; }

  bool is_maia() const noexcept { return false; }

  bool is_meta() const noexcept { return false; }

  bool is_cpu() const noexcept { return type_ == DeviceType::CPU; }

  bool supports_as_strided() const noexcept { return type_ != DeviceType::IPU; }

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
        return has_index() ? phi::GPUPlace(index_) : paddle::DefaultGPUPlace();
      case DeviceType::XPU:
        return has_index() ? phi::XPUPlace(index_) : paddle::DefaultXPUPlace();
      case DeviceType::IPU:
        return has_index() ? phi::IPUPlace(index_) : phi::IPUPlace();
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

  void validate() {
#ifndef NDEBUG
    TORCH_INTERNAL_ASSERT(index_ >= -1,
                          "Device index must be -1 or non-negative, got ",
                          static_cast<int>(index_));
    TORCH_INTERNAL_ASSERT(!is_cpu() || index_ <= 0,
                          "CPU device index must be -1 or zero, got ",
                          static_cast<int>(index_));
#endif
  }
};

PADDLE_API std::ostream& operator<<(std::ostream& stream, const Device& device);

}  // namespace c10

namespace std {
template <>
struct hash<c10::Device> {
  size_t operator()(c10::Device d) const noexcept {
    static_assert(sizeof(c10::DeviceType) == 1, "DeviceType is not 8-bit");
    static_assert(sizeof(c10::DeviceIndex) == 1, "DeviceIndex is not 8-bit");
    uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(d.type()))
                        << 16 |
                    static_cast<uint32_t>(static_cast<uint8_t>(d.index()));
    return std::hash<uint32_t>{}(bits);
  }
};
}  // namespace std

namespace at {
using c10::Device;
using c10::DeviceIndex;
}  // namespace at

namespace torch {
using c10::Device;
using c10::DeviceIndex;
}  // namespace torch
