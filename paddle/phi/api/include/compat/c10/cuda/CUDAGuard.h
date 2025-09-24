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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/core/Device.h>
#include "paddle/phi/core/platform/cuda_device_guard.h"

namespace c10::cuda {
struct CUDAGuard {
  explicit CUDAGuard() = delete;  // NOLINT

  explicit CUDAGuard(DeviceIndex device_index) : guard_(device_index) {}

  explicit CUDAGuard(Device device) : guard_(device._PD_GetInner()) {}

  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  CUDAGuard(CUDAGuard&& other) = delete;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;
  ~CUDAGuard() = default;

  void set_device(Device device) { guard_.SetDevice(device._PD_GetInner()); }

  void reset_device(Device device) { set_device(device); }

  void set_index(DeviceIndex device_index) {
    guard_.SetDeviceIndex(device_index);
  }

  Device current_device() const {
    return c10::Device(c10::kCUDA, phi::backends::gpu::GetCurrentDeviceId());
  }

 private:
  paddle::platform::CUDADeviceGuard guard_;
};

struct OptionalCUDAGuard {
  OptionalCUDAGuard() = default;

  explicit OptionalCUDAGuard(std::optional<Device> device_opt) : guard_() {
    if (device_opt.has_value()) {
      guard_.emplace(device_opt.value()._PD_GetInner());
    }
  }

  explicit OptionalCUDAGuard(std::optional<DeviceIndex> device_index_opt)
      : guard_() {
    if (device_index_opt.has_value()) {
      guard_.emplace(device_index_opt.value());
    }
  }

  // Copy is not allowed
  OptionalCUDAGuard(const OptionalCUDAGuard&) = delete;
  OptionalCUDAGuard& operator=(const OptionalCUDAGuard&) = delete;

  OptionalCUDAGuard(OptionalCUDAGuard&& other) = delete;

  OptionalCUDAGuard& operator=(OptionalCUDAGuard&& other) = delete;
  ~OptionalCUDAGuard() = default;

  void set_device(Device device) {
    if (!guard_.has_value()) {
      guard_.emplace(device._PD_GetInner());
    } else {
      guard_->SetDevice(device._PD_GetInner());
    }
  }

  void reset_device(Device device) {
    if (!guard_.has_value()) {
      guard_.emplace(device._PD_GetInner());
    } else {
      guard_->SetDevice(device._PD_GetInner());
    }
  }

  void set_index(DeviceIndex device_index) {
    if (!guard_.has_value()) {
      guard_.emplace(device_index);
    } else {
      guard_->SetDeviceIndex(device_index);
    }
  }

  std::optional<Device> current_device() const {
    return guard_.has_value()
               ? std::make_optional(c10::Device(
                     c10::kCUDA, phi::backends::gpu::GetCurrentDeviceId()))
               : std::nullopt;
  }

 private:
  std::optional<paddle::platform::CUDADeviceGuard> guard_;
};

}  // namespace c10::cuda

namespace at::cuda {
using c10::cuda::CUDAGuard;
using c10::cuda::OptionalCUDAGuard;
}  // namespace at::cuda
