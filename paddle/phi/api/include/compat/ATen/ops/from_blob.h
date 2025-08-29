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
#include <ATen/core/Tensor.h>

#include "paddle/phi/api/include/tensor_utils.h"
namespace at {

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = std::nullopt) {
  return paddle::from_blob(
      data,
      sizes._PD_ToPaddleIntArray(),
      strides._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      phi::DataLayout::NCHW,
      device_or_default(target_device)._PD_GetInner(),
      deleter);
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t storage_offset,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = std::nullopt) {
  PD_CHECK(storage_offset == 0, "`storage_offset` should be zero.");

  return paddle::from_blob(
      data,
      sizes._PD_ToPaddleIntArray(),
      strides._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      phi::DataLayout::NCHW,
      device_or_default(target_device)._PD_GetInner(),
      deleter);
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = std::nullopt) {
  return paddle::from_blob(
      data,
      sizes._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      phi::DataLayout::NCHW,
      device_or_default(target_device)._PD_GetInner(),
      deleter);
}

inline Tensor from_blob(void* data,
                        IntArrayRef sizes,
                        IntArrayRef strides,
                        const TensorOptions& options = {}) {
  return paddle::from_blob(
      data,
      sizes._PD_ToPaddleIntArray(),
      strides._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      phi::DataLayout::NCHW,
      options._PD_GetPlace());
}

inline Tensor from_blob(void* data,
                        IntArrayRef sizes,
                        const TensorOptions& options = {}) {
  return paddle::from_blob(
      data,
      sizes._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      phi::DataLayout::NCHW,
      options._PD_GetPlace(),
      nullptr);
}

}  // namespace at
namespace torch {
using at::from_blob;
}  // namespace torch
