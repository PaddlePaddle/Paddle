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
#include <c10/core/TensorOptions.h>
#include <optional>

#include "paddle/phi/api/include/api.h"

namespace at {

inline at::Tensor arange(const at::Scalar& end,
                         at::TensorOptions options = {}) {
  return paddle::experimental::arange(
      paddle::experimental::full({}, 0, phi::DataType::FLOAT64),
      paddle::experimental::full({}, end.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full({}, 1, phi::DataType::FLOAT64),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());
}

inline at::Tensor arange(const at::Scalar& end,
                         ::std::optional<at::ScalarType> dtype,
                         ::std::optional<at::Layout> layout,
                         ::std::optional<at::Device> device,
                         ::std::optional<bool> pin_memory) {
  return paddle::experimental::arange(
      paddle::experimental::full({}, 0, phi::DataType::FLOAT64),
      paddle::experimental::full({}, end.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full({}, 1, phi::DataType::FLOAT64),
      compat::_PD_AtenScalarTypeToPhiDataType(
          dtype.value_or(c10::get_default_dtype())),
      device.value_or(at::kCPU)._PD_GetInner());
}

inline at::Tensor arange(const at::Scalar& start,
                         const at::Scalar& end,
                         at::TensorOptions options = {}) {
  return paddle::experimental::arange(
      paddle::experimental::full(
          {}, start.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full({}, end.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full({}, 1, phi::DataType::FLOAT64),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());
}

inline at::Tensor arange(const at::Scalar& start,
                         const at::Scalar& end,
                         ::std::optional<at::ScalarType> dtype,
                         ::std::optional<at::Layout> layout,
                         ::std::optional<at::Device> device,
                         ::std::optional<bool> pin_memory) {
  return paddle::experimental::arange(
      paddle::experimental::full(
          {}, start.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full({}, end.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full({}, 1, phi::DataType::FLOAT64),
      compat::_PD_AtenScalarTypeToPhiDataType(
          dtype.value_or(c10::get_default_dtype())),
      device.value_or(at::kCPU)._PD_GetInner());
}

inline at::Tensor arange(const at::Scalar& start,
                         const at::Scalar& end,
                         const at::Scalar& step,
                         at::TensorOptions options = {}) {
  return paddle::experimental::arange(
      paddle::experimental::full(
          {}, start.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full({}, end.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full(
          {}, step.to<int64_t>(), phi::DataType::FLOAT64),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());
}

inline at::Tensor arange(const at::Scalar& start,
                         const at::Scalar& end,
                         const at::Scalar& step,
                         ::std::optional<at::ScalarType> dtype,
                         ::std::optional<at::Layout> layout,
                         ::std::optional<at::Device> device,
                         ::std::optional<bool> pin_memory) {
  return paddle::experimental::arange(
      paddle::experimental::full(
          {}, start.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full({}, end.to<int64_t>(), phi::DataType::FLOAT64),
      paddle::experimental::full(
          {}, step.to<int64_t>(), phi::DataType::FLOAT64),
      compat::_PD_AtenScalarTypeToPhiDataType(
          dtype.value_or(c10::get_default_dtype())),
      device.value_or(at::kCPU)._PD_GetInner());
}

}  // namespace at

namespace torch {
using at::arange;
}  // namespace torch
