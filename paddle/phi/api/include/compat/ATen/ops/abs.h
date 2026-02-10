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
#include <string_view>

#include "paddle/phi/api/include/api.h"

namespace at {

inline at::Tensor abs(const at::Tensor& self) {
  return paddle::experimental::abs(self._PD_GetInner());
}

inline at::Tensor& abs_(at::Tensor& self) {  // NOLINT(runtime/references)
  paddle::experimental::abs_(self._PD_GetInner());
  return self;
}

// Tensor member function implementations
inline at::Tensor Tensor::abs() const { return at::abs(*this); }

inline at::Tensor& Tensor::abs_() const {
  return at::abs_(const_cast<at::Tensor&>(*this));
}

}  // namespace at

namespace torch {
using at::abs;
using at::abs_;
}  // namespace torch
