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
#include <c10/util/ArrayRef.h>
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"

namespace at {

inline at::Tensor empty_strided(at::IntArrayRef size,
                                at::IntArrayRef stride,
                                at::TensorOptions options = {}) {
  auto empty_tensor = paddle::experimental::empty(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());

  return paddle::experimental::as_strided(
      empty_tensor,
      std::vector<int64_t>(size.begin(), size.end()),
      std::vector<int64_t>(stride.begin(), stride.end()));
}

}  // namespace at

namespace torch {
using at::empty_strided;
}  // namespace torch
