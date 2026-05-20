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
#include "paddle/phi/core/enforce.h"

namespace at {

inline at::Tensor abs(const at::Tensor& self) {
  if (!self.is_contiguous()) {
    phi::enforce::ThrowWarnInternal(
        "at::abs: input tensor is non-contiguous. PyTorch and Paddle handle "
        "non-contiguous tensors differently, which may produce logically "
        "incorrect results even though the code is syntactically valid. "
        "See https://github.com/PaddlePaddle/Paddle/pull/78099 for details.");
  }
  return paddle::experimental::abs(self._PD_GetInner());
}

}  // namespace at

namespace at {

inline at::Tensor Tensor::abs() const { return at::abs(*this); }

inline at::Tensor& Tensor::abs_() const {
  if (!is_contiguous()) {
    phi::enforce::ThrowWarnInternal(
        "Tensor::abs_: tensor is non-contiguous. PyTorch and Paddle handle "
        "non-contiguous tensors differently, which may produce logically "
        "incorrect results even though the code is syntactically valid. "
        "See https://github.com/PaddlePaddle/Paddle/pull/78099 for details.");
  }
  PaddleTensor& inner = const_cast<PaddleTensor&>(tensor_);
  paddle::experimental::abs_(inner);
  return const_cast<at::Tensor&>(*this);
}

}  // namespace at
