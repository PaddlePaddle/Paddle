// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <tuple>

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"

namespace paddle {
namespace experimental {

PADDLE_API std::vector<std::vector<Tensor>> matmul_grad(
    const Tensor& x,
    const Tensor& y,
    const Tensor& out_grad,
    bool transpose_x = false,
    bool transpose_y = false);

PADDLE_API Tensor scale_grad(const Tensor& out_grad,
                             const Scalar& scale,
                             float bias = 0.0,
                             bool bias_after_scale = true);

}  // namespace experimental
}  // namespace paddle
