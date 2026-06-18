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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <optional>

#include "paddle/phi/backends/gpu/gpu_info.h"

namespace c10::cuda {

namespace detail {

inline Device current_cuda_device() {
  return Device(kCUDA, phi::backends::gpu::GetCurrentDeviceId());
}

inline Device normalize_cuda_device(Device device) {
  TORCH_CHECK(device.is_cuda(), "Expected a CUDA device, but got ", device);
  return device.has_index() ? Device(kCUDA, device.index())
                            : current_cuda_device();
}

}  // namespace detail

struct CUDAGuard {
  explicit CUDAGuard() = delete;  // NOLINT

  explicit CUDAGuard(DeviceIndex device_index)
      : original_device_(detail::current_cuda_device()),
        current_device_(original_device_) {
    set_index(device_index);
  }

  explicit CUDAGuard(Device device)
      : original_device_(detail::current_cuda_device()),
        current_device_(original_device_) {
    set_device(device);
  }

  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  CUDAGuard(CUDAGuard&& other) = delete;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;
  ~CUDAGuard() {
    // Always restore to original_device_ to handle cases where the device
    // was changed outside of this guard, matching PyTorch semantics.
    phi::backends::gpu::SetDeviceId(static_cast<int>(original_device_.index()));
  }

  void set_device(Device device) {
    const Device normalized = detail::normalize_cuda_device(device);
    if (normalized.index() != current_device_.index()) {
      phi::backends::gpu::SetDeviceId(static_cast<int>(normalized.index()));
      current_device_ = normalized;
    }
  }

  void reset_device(Device device) { set_device(device); }

  void set_index(DeviceIndex device_index) {
    if (current_device_.index() != device_index) {
      phi::backends::gpu::SetDeviceId(static_cast<int>(device_index));
      current_device_ = Device(kCUDA, device_index);
    }
  }

  Device original_device() const { return original_device_; }

  Device current_device() const { return current_device_; }

 private:
  Device original_device_;
  Device current_device_;
};

struct OptionalCUDAGuard {
  OptionalCUDAGuard() = default;

  explicit OptionalCUDAGuard(std::optional<Device> device_opt) {
    if (device_opt.has_value()) {
      set_device(device_opt.value());
    }
  }

  explicit OptionalCUDAGuard(std::optional<DeviceIndex> device_index_opt) {
    if (device_index_opt.has_value()) {
      set_index(device_index_opt.value());
    }
  }

  OptionalCUDAGuard(const OptionalCUDAGuard&) = delete;
  OptionalCUDAGuard& operator=(const OptionalCUDAGuard&) = delete;

  OptionalCUDAGuard(OptionalCUDAGuard&& other) = delete;
  OptionalCUDAGuard& operator=(OptionalCUDAGuard&& other) = delete;
  ~OptionalCUDAGuard() { reset(); }

  void set_device(Device device) {
    const Device normalized = detail::normalize_cuda_device(device);
    init_if_needed();
    if (normalized.index() != current_device_->index()) {
      phi::backends::gpu::SetDeviceId(static_cast<int>(normalized.index()));
    }
    current_device_ = normalized;
  }

  void reset_device(Device device) { set_device(device); }

  void set_index(DeviceIndex device_index) {
    init_if_needed();
    if (device_index != current_device_->index()) {
      phi::backends::gpu::SetDeviceId(static_cast<int>(device_index));
    }
    current_device_ = Device(kCUDA, device_index);
  }

  std::optional<Device> original_device() const { return original_device_; }

  std::optional<Device> current_device() const { return current_device_; }

  void reset() {
    if (original_device_.has_value()) {
      // Always restore to original_device_ to handle external device changes.
      // This matches PyTorch OptionalDeviceGuard semantics.
      phi::backends::gpu::SetDeviceId(
          static_cast<int>(original_device_->index()));
    }
    original_device_.reset();
    current_device_.reset();
  }

 private:
  void init_if_needed() {
    if (!original_device_.has_value()) {
      original_device_ = detail::current_cuda_device();
      current_device_ = original_device_;
    }
  }

  std::optional<Device> original_device_;
  std::optional<Device> current_device_;
};

}  // namespace c10::cuda

namespace at::cuda {
using c10::cuda::CUDAGuard;
using c10::cuda::OptionalCUDAGuard;
}  // namespace at::cuda
