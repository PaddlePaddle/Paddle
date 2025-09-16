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

#include <c10/core/TensorOptions.h>
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"

namespace at {

inline at::Tensor sum(const at::Tensor& self,
                      ::std::optional<at::ScalarType> dtype = ::std::nullopt) {
  return paddle::experimental::sum(
      self._PD_GetInner(),
      {},
      compat::_PD_AtenScalarTypeToPhiDataType(
          dtype.value_or(c10::get_default_dtype())),
      /*keepdim=*/false);
}

inline at::Tensor sum(const at::Tensor& self,
                      at::OptionalIntArrayRef dim,
                      bool keepdim = false,
                      ::std::optional<at::ScalarType> dtype = ::std::nullopt) {
  return paddle::experimental::sum(
      self._PD_GetInner(),
      dim.has_value() ? dim.value()._PD_ToPaddleIntArray()
                      : paddle::experimental::IntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(
          dtype.value_or(c10::get_default_dtype())),
      keepdim);
}

inline at::Tensor& sum_out(
    at::Tensor&
        out,  // NOLINT: intentional non-const reference for output parameter
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim = false,
    ::std::optional<at::ScalarType> dtype = ::std::nullopt) {
  auto res = sum(self, dim, keepdim, dtype);
  paddle::experimental::assign_out_(res._PD_GetInner(), out._PD_GetInner());
  return out;
}

inline at::Tensor& sum_out(
    at::Tensor&
        out,  // NOLINT: intentional non-const reference for output parameter
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype = ::std::nullopt) {
  auto res = sum(self, dtype);
  paddle::experimental::assign_out_(res._PD_GetInner(), out._PD_GetInner());
  return out;
}

}  // namespace at

namespace torch {
using at::sum;
using at::sum_out;
}  // namespace torch
