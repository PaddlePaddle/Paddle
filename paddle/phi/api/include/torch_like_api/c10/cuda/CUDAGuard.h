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
#include <c10/core/Device.h>
#include "paddle/phi/core/platform/cuda_device_guard.h"

namespace c10::cuda {
struct CUDAGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit CUDAGuard() = delete;  // NOLINT

  /// Set the current CUDA device to the passed device index.
  explicit CUDAGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current CUDA device to the passed device.  Errors if the passed
  /// device is not a CUDA device.
  explicit CUDAGuard(Device device) : guard_(device._PD_GetInner()) {}

  // Copy is not allowed
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
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

/// A variant of OptionalDeviceGuard that is specialized for CUDA.  See
/// CUDAGuard for when you can use this.
struct OptionalCUDAGuard {
  /// Create an uninitialized OptionalCUDAGuard.
  OptionalCUDAGuard() = default;

  /// Set the current CUDA device to the passed Device, if it is not nullopt.
  explicit OptionalCUDAGuard(std::optional<Device> device_opt) : guard_() {
    if (device_opt.has_value()) {
      guard_.emplace(device_opt.value()._PD_GetInner());
    }
  }

  /// Set the current CUDA device to the passed device index, if it is not
  /// nullopt
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

  /// Sets the CUDA device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a CUDA device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) {
    if (!guard_.has_value()) {
      guard_.emplace(device._PD_GetInner());
    } else {
      guard_->SetDevice(device._PD_GetInner());
    }
  }

  /// Sets the CUDA device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) {
    if (!guard_.has_value()) {
      guard_.emplace(device_index);
    } else {
      guard_->SetDeviceIndex(device_index);
    }
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
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
using c10::cuda;
}  // namespace at::cuda
