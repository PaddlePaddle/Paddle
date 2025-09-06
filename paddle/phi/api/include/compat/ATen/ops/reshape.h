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

inline at::Tensor reshape(const at::Tensor& self, at::IntArrayRef shape) {
  return paddle::experimental::reshape(self._PD_GetInner(),
                                       shape._PD_ToPaddleIntArray());
}

inline at::Tensor reshape_symint(const at::Tensor& self,
                                 c10::SymIntArrayRef shape) {
  return paddle::experimental::reshape(self._PD_GetInner(),
                                       shape._PD_ToPaddleIntArray());
}

}  // namespace at
namespace torch {
using at::reshape;
using at::reshape_symint;
}  // namespace torch
