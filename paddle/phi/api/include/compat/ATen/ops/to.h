// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <utils/scalar_type_conversion.h>
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

namespace at {

// Overload 1: to(TensorOptions, non_blocking, copy, memory_format)
inline at::Tensor Tensor::to(
    at::TensorOptions options,
    bool non_blocking,
    bool copy,
    ::std::optional<at::MemoryFormat> memory_format) const {
  // Handle device transfer
  PaddleTensor result = tensor_;
  bool materialized_copy = false;

  if (options.has_device()) {
    const c10::Device& dev = options.device();
    phi::Place place;
    switch (dev.type()) {
      case c10::DeviceType::CPU:
        place = phi::CPUPlace();
        break;
      case c10::DeviceType::CUDA:
        place = dev.has_index() ? phi::GPUPlace(dev.index())
                                : paddle::DefaultGPUPlace();
        break;
      default:
        PD_THROW("Unsupported device type: ", dev.type());
        break;
    }
    if (place != tensor_.place()) {
      result = result.copy_to(place, /*blocking=*/!non_blocking);
      materialized_copy = true;
    }
  }

  // Handle dtype cast
  if (options.has_dtype()) {
    auto target_dtype =
        compat::_PD_AtenScalarTypeToPhiDataType(options.dtype());
    if (target_dtype != result.dtype()) {
      result = paddle::experimental::cast(result, target_dtype);
      materialized_copy = true;
    }
  }

  if (copy && !materialized_copy) {
    result = paddle::experimental::assign(result);
  }

  return at::Tensor(result);
}

// Overload 2: to(optional<ScalarType>, optional<Layout>, optional<Device>,
//               optional<bool> pin_memory, non_blocking, copy, memory_format)
inline at::Tensor Tensor::to(
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    ::std::optional<at::MemoryFormat> memory_format) const {
  at::TensorOptions options;
  if (dtype.has_value()) {
    options = options.dtype(dtype.value());
  }
  if (device.has_value()) {
    options = options.device(device.value());
  }
  if (pin_memory.has_value() && pin_memory.value()) {
    options = options.pinned_memory(true);
  }
  return to(options, non_blocking, copy, memory_format);
}

// Overload 3: to(Device, ScalarType, non_blocking, copy, memory_format)
inline at::Tensor Tensor::to(
    at::Device device,
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    ::std::optional<at::MemoryFormat> memory_format) const {
  at::TensorOptions options = at::TensorOptions().device(device).dtype(dtype);
  return to(options, non_blocking, copy, memory_format);
}

// Overload 4: to(ScalarType, non_blocking, copy, memory_format)
inline at::Tensor Tensor::to(
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    ::std::optional<at::MemoryFormat> memory_format) const {
  auto target_dtype = compat::_PD_AtenScalarTypeToPhiDataType(dtype);
  if (!copy && target_dtype == tensor_.dtype()) {
    return *this;
  }
  return at::Tensor(paddle::experimental::cast(tensor_, target_dtype));
}

// Overload 5: to(const Tensor& other, non_blocking, copy, memory_format)
inline at::Tensor Tensor::to(
    const at::Tensor& other,
    bool non_blocking,
    bool copy,
    ::std::optional<at::MemoryFormat> memory_format) const {
  at::TensorOptions options =
      at::TensorOptions().device(other.device()).dtype(other.scalar_type());
  return to(options, non_blocking, copy, memory_format);
}

}  // namespace at
