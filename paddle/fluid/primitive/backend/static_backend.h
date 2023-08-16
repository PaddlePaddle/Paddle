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
std::tuple<Tensor, Tensor> add_grad(const Tensor& x,
                                    const Tensor& y,
                                    const Tensor& out_grad,
                                    int axis);

template <typename T>
Tensor divide(const Tensor& x, const Tensor& y);

template <typename T>
Tensor add(const Tensor& x, const Tensor& y);

template <typename T>
Tensor multiply(const Tensor& x, const Tensor& y);

template <typename T>
Tensor elementwise_pow(const Tensor& x, const Tensor& y);

template <typename T>
Tensor scale(const Tensor& x,
             const Scalar& scale = 1.0,
             float bias = 0.0,
             bool bias_after_scale = true);

template <typename T>
Tensor sum(const Tensor& x,
           const IntArray& axis = {},
           phi::DataType dtype = phi::DataType::UNDEFINED,
           bool keepdim = false);

template <typename T>
Tensor full(const IntArray& shape,
            const Scalar& value,
            phi::DataType dtype = phi::DataType::FLOAT32,
            phi::Place place = phi::CPUPlace());

template <typename T>
std::tuple<Tensor, Tensor> reshape(const Tensor& x, const IntArray& shape);

template <typename T>
Tensor expand(const Tensor& x, const IntArray& shape);

template <typename T>
Tensor tile(const Tensor& x, const IntArray& repeat_times = {});

}  // namespace backend
}  // namespace primitive
}  // namespace paddle
