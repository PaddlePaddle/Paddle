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

namespace paddle {
namespace primitive {
namespace backend {

using Tensor = paddle::Tensor;
using Scalar = paddle::experimental::Scalar;
using IntArray = paddle::experimental::IntArray;
using DataType = phi::DataType;

template <typename T>
std::vector<Tensor> add_n_grad(const std::vector<Tensor>& x,
                               const Tensor& out_grad);

}  // namespace backend
}  // namespace primitive
}  // namespace paddle
