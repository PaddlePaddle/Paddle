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
#include <c10/core/Scalar.h>
#include <c10/util/OptionalArrayRef.h>

#include "paddle/phi/api/include/api.h"

namespace at {

// any - returns true if any element is non-zero
inline Tensor any(const Tensor& self, int64_t dim, bool keepdim = false) {
  auto result = paddle::experimental::sum(
      self._PD_GetInner(), phi::IntArray({dim}), phi::DataType::BOOL, keepdim);
  return Tensor(paddle::experimental::cast(result, phi::DataType::BOOL));
}

inline Tensor any(const Tensor& self,
                  at::OptionalIntArrayRef dim,
                  bool keepdim = false) {
  std::vector<int64_t> dims_vec;
  if (dim.has_value() && dim.value().size() > 0) {
    dims_vec.assign(dim.value().begin(), dim.value().end());
  }
  auto result = paddle::experimental::sum(self._PD_GetInner(),
                                          phi::IntArray(dims_vec),
                                          phi::DataType::BOOL,
                                          keepdim);
  return Tensor(paddle::experimental::cast(result, phi::DataType::BOOL));
}

inline Tensor any(const Tensor& self) {
  auto result = paddle::experimental::sum(self._PD_GetInner(),
                                          phi::IntArray(std::vector<int64_t>{}),
                                          phi::DataType::BOOL,
                                          false);
  return Tensor(paddle::experimental::cast(result, phi::DataType::BOOL));
}

}  // namespace at
