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

#include <ATen/common.h>
#include <ATen/core/TensorBody.h>
#include "paddle/phi/api/include/api.h"

namespace at {

inline at::Tensor empty(
    at::IntArrayRef size,
    at::TensorOptions options = {},
    ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  if (memory_format.has_value()) {
    UNSUPPORTED_FEATURE_IN_PADDLE("`MemoryFormat`")
  }
  return paddle::experimental::empty(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options._PD_GetScalarType()),
      options._PD_GetPlace());
}
inline at::Tensor empty_like(
    const at::Tensor& self,
    at::TensorOptions options = {},
    ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  if (memory_format.has_value()) {
    UNSUPPORTED_FEATURE_IN_PADDLE("`MemoryFormat`")
  }
  auto dtype = options.dtype_opt().value_or(self.dtype());
  auto place = options.device_opt().value_or(self.device());
  return paddle::experimental::empty_like(
      self._PD_GetInner(),
      compat::_PD_AtenScalarTypeToPhiDataType(dtype),
      place._PD_GetInner());
}
inline at::Tensor ones(at::IntArrayRef size, at::TensorOptions options = {}) {
  return paddle::experimental::ones(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options._PD_GetScalarType()),
      options._PD_GetPlace());
}
inline at::Tensor zeros(at::IntArrayRef size, at::TensorOptions options = {}) {
  return paddle::experimental::zeros(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options._PD_GetScalarType()),
      options._PD_GetPlace());
}
inline at::Tensor zeros_like(
    const at::Tensor& self,
    at::TensorOptions options = {},
    ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  if (memory_format.has_value()) {
    UNSUPPORTED_FEATURE_IN_PADDLE("`MemoryFormat`")
  }
  auto dtype = options.dtype_opt().value_or(self.dtype());
  auto place = options.device_opt().value_or(self.device());
  return paddle::experimental::zeros_like(
      self._PD_GetInner(),
      compat::_PD_AtenScalarTypeToPhiDataType(dtype),
      place._PD_GetInner());
}
inline at::Tensor full(at::IntArrayRef size,
                       const at::Scalar& fill_value,
                       ::std::optional<at::ScalarType> dtype = {},
                       ::std::optional<at::Layout> layout = {},
                       ::std::optional<at::Device> device = {},
                       ::std::optional<bool> pin_memory = {}) {
  if (pin_memory.has_value()) {
    UNSUPPORTED_FEATURE_IN_PADDLE("`pin_memory` option in full")
  }
  return paddle::experimental::full(
      size._PD_ToPaddleIntArray(),
      fill_value,
      dtype.has_value() ? compat::_PD_AtenScalarTypeToPhiDataType(*dtype)
                        : phi::DataType::FLOAT32,
      device.has_value() ? device.value()._PD_GetInner() : phi::CPUPlace());
}
inline at::Tensor abs(const at::Tensor& self) {
  return paddle::experimental::abs(self._PD_GetInner());
}

inline at::Tensor sum(const at::Tensor& self,
                      ::std::optional<at::ScalarType> dtype = ::std::nullopt) {
  return paddle::experimental::sum(
      self._PD_GetInner(),
      {},
      dtype.has_value() ? compat::_PD_AtenScalarTypeToPhiDataType(dtype.value())
                        : phi::DataType::UNDEFINED,
      /*keepdim=*/false);
}
inline at::Tensor& sum_out(
    at::Tensor& out,  // NOLINT
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype = ::std::nullopt) {
  auto res = sum(self, dtype);
  paddle::experimental::assign_out_(res._PD_GetInner(), out._PD_GetInner());
  return out;
}

inline at::Tensor sum(const at::Tensor& self,
                      at::OptionalIntArrayRef dim,
                      bool keepdim = false,
                      ::std::optional<at::ScalarType> dtype = ::std::nullopt) {
  return paddle::experimental::sum(
      self._PD_GetInner(),
      dim.has_value() ? dim.value()._PD_ToPaddleIntArray()
                      : paddle::experimental::IntArray(),
      dtype.has_value() ? compat::_PD_AtenScalarTypeToPhiDataType(dtype.value())
                        : phi::DataType::UNDEFINED,
      keepdim);
}
inline at::Tensor& sum_out(
    at::Tensor& out,  // NOLINT
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim = false,
    ::std::optional<at::ScalarType> dtype = ::std::nullopt) {
  auto res = sum(self, dim, keepdim, dtype);
  paddle::experimental::assign_out_(res._PD_GetInner(), out._PD_GetInner());
  return out;
}

inline at::Tensor reshape(const at::Tensor& self, at::IntArrayRef shape) {
  return paddle::experimental::reshape(self._PD_GetInner(),
                                       shape._PD_ToPaddleIntArray());
}

}  // namespace at
