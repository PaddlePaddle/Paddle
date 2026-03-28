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
  Device(phi::Place place)
      : inner_(place),
        has_index_(
            phiPlaceHasC10DeviceIndex(place.GetType(), place.GetDeviceId())) {}

  // PyTorch 兼容: Device(DeviceType, DeviceIndex)
  // GPUPINNED / XPUPINNED 为无索引设备，index 参数将被忽略。
  // 其余设备（含 CPU）index=-1 表示未指定。
  Device(DeviceType type, DeviceIndex index = -1) {
    const phi::AllocationType alloc = c10DeviceTypeToPhiAllocationType(type);
    // 仅固定内存设备无 index 语义。
    // 其它设备遵循 PyTorch: index==-1 表示未显式指定。
    const bool no_index_type =
        (type == DeviceType::GPUPINNED || type == DeviceType::XPUPINNED);
    has_index_ = !no_index_type && (index != -1);
    inner_ = phi::Place(alloc, has_index_ ? index : 0);
  }

  // PyTorch 兼容: Device(DeviceType::CUSTOM, index, custom_device_type)
  Device(DeviceType type,
         DeviceIndex index,
         const std::string& custom_device_type) {
    const phi::AllocationType alloc = c10DeviceTypeToPhiAllocationType(type);
    const bool no_index_type =
        (type == DeviceType::GPUPINNED || type == DeviceType::XPUPINNED);
    has_index_ = !no_index_type && (index != -1);

    if (type == DeviceType::CUSTOM) {
      inner_ =
          phi::Place(alloc,
                     has_index_ ? index : 0,
                     custom_device_type.empty() ? std::string("privateuseone")
                                                : custom_device_type);
    } else {
      inner_ = phi::Place(alloc, has_index_ ? index : 0);
    }
  }

  /// Constructs a `Device` from a string description, for convenience.
  /// Supported formats: `(cpu|cuda|xpu|ipu|custom)[:<device-index>]`
  /// e.g. "cuda:0", "xpu:1", "cpu", "custom:2"
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

  // PyTorch 兼容: is_cuda() 检查底层是否为 GPU（phi::AllocationType::GPU）
  bool is_cuda() const noexcept {
    return inner_.GetType() == phi::AllocationType::GPU;
  }

  bool is_cpu() const noexcept {
    return inner_.GetType() == phi::AllocationType::CPU;
  }

  bool is_xpu() const noexcept {
    return inner_.GetType() == phi::AllocationType::XPU;
  }

  bool is_ipu() const noexcept {
    return inner_.GetType() == phi::AllocationType::IPU;
  }

  bool is_custom() const noexcept {
    return inner_.GetType() == phi::AllocationType::CUSTOM;
  }

  // 判断是否为固定内存设备（GPUPINNED 或 XPUPINNED）
  bool is_pinned() const noexcept {
    const auto t = inner_.GetType();
    return t == phi::AllocationType::GPUPINNED ||
           t == phi::AllocationType::XPUPINNED;
  }

  std::string str() const;

  bool operator==(const Device& other) const noexcept {
    return inner_ == other.inner_ && has_index_ == other.has_index_;
  }

  phi::Place _PD_GetInner() const { return inner_; }

 private:
  phi::Place inner_;
  bool has_index_{false};
};

std::ostream& operator<<(std::ostream& stream, const Device& device);

// Parse device type string to DeviceType
DeviceType parse_type(const std::string& device_string);

}  // namespace c10

namespace at {
using c10::Device;
using c10::DeviceIndex;
}  // namespace at

namespace torch {
using c10::Device;
using c10::DeviceIndex;
}  // namespace torch
