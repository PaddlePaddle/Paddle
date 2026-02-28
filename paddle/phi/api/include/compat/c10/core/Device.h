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
#include <c10/core/DeviceType.h>

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
#endif

#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/device_event_base.h"

namespace c10 {
using DeviceIndex = int8_t;

struct Device final {
  using Type = DeviceType;
  Device(phi::Place place)
      : inner_(place),
        has_index_(
            phiPlaceHasC10DeviceIndex(place.GetType(), place.GetDeviceId())) {}

  // PyTorch 兼容: Device(DeviceType, DeviceIndex)
  // 注意：CPU 默认 index = -1，但转换为 phi::Place 时使用 0 以保持兼容性
  // CUDA/GPU 默认 index = -1
  Device(DeviceType type, DeviceIndex index = -1)
      : inner_(c10DeviceTypeToPhiAllocationType(type),
               type == DeviceType::CPU ? 0 : index),  // CPU 始终使用 device=0
        has_index_(type == DeviceType::CPU ? false : (index != -1)) {}

  /// Constructs a `Device` from a string description, for convenience.
  /// The string supplied must follow the following schema:
  /// `(cpu|cuda)[:<device-index>]`
  /// where `cpu` or `cuda` specifies the device type, and
  /// `:<device-index>` optionally specifies a device index.
  /* implicit */ Device(const std::string& device_string);

  DeviceIndex index() const noexcept {
    return has_index_ ? inner_.GetDeviceId() : -1;
  }

  // PyTorch 兼容: has_index() = (index != -1)
  bool has_index() const noexcept { return has_index_; }

  // 返回与 PyTorch 兼容的 DeviceType
  DeviceType type() const {
    return phiAllocationTypeToC10DeviceType(inner_.GetType());
  }

  // PyTorch 兼容: is_cuda() 检查 CUDA 和 GPU 类型
  bool is_cuda() const noexcept {
    auto t = inner_.GetType();
    return t == phi::AllocationType::GPU;
  }

  bool is_cpu() const noexcept {
    return inner_.GetType() == phi::AllocationType::CPU;
  }

  std::string str() const;

  bool operator==(const Device& other) const noexcept {
    return type() == other.type() && this->index() == other.index();
  }

  phi::Place _PD_GetInner() const { return inner_; }

 private:
  phi::Place inner_;
  bool has_index_{false};
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
