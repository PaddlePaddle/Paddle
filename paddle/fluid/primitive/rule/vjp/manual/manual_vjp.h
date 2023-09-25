// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/pir/core/value.h"

namespace paddle {
namespace primitive {

using IntArray = paddle::experimental::IntArray;
std::vector<std::vector<paddle::Tensor>> add_n_vjp(
    const std::vector<paddle::Tensor>& x,
    const Tensor& out_grad,
    const std::vector<std::vector<bool>>& stop_gradients);
}  // namespace primitive
}  // namespace paddle
