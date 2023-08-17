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

#include <string>
#include <vector>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"

namespace paddle {
namespace primitive {
namespace backend {
namespace experimental {

using Tensor = paddle::Tensor;
using IntArray = paddle::experimental::IntArray;

template <typename T>
Tensor tanh_grad(const Tensor& out, const Tensor& grad_out);

template <typename T>
Tensor mean_grad(const Tensor& x,
                 const Tensor& out_grad,
                 const IntArray& axis = {},
                 bool keepdim = false,
                 bool reduce_all = false);

template <typename T>
std::vector<Tensor> concat_grad(const std::vector<Tensor>& x,
                                const Tensor& out_grad,
                                const Tensor& axis);

template <typename T>
std::tuple<Tensor, Tensor> add_grad(const Tensor& x,
                                    const Tensor& y,
                                    const Tensor& out_grad,
                                    int axis);
}  // namespace experimental
}  // namespace backend
}  // namespace primitive
}  // namespace paddle
