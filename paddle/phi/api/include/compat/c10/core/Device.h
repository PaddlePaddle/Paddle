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
  Device(phi::Place place) : inner_(place) {}
  Device(DeviceType type, DeviceIndex index = 0)
      : inner_(phi::Place(type, index)) {}  // NOLINT

  /// Constructs a `Device` from a string description, for convenience.
  /// The string supplied must follow the following schema:
  /// `(cpu|cuda)[:<device-index>]`
  /// where `cpu` or `cuda` specifies the device type, and
  /// `:<device-index>` optionally specifies a device index.
  /* implicit */ Device(const std::string& device_string);

  DeviceIndex index() const noexcept { return inner_.GetDeviceId(); }

  bool has_index() const noexcept { return index() != -1; }

  DeviceType type() const { return inner_.GetType(); }

  bool is_cuda() const noexcept { return phi::is_gpu_place(inner_); }

  bool is_cpu() const noexcept { return phi::is_cpu_place(inner_); }

  std::string str() const;

  bool operator==(const Device& other) const noexcept {
    return type() == other.type() && this->index() == other.index();
  }

  phi::Place _PD_GetInner() const { return inner_; }

 private:
  phi::Place inner_;
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
