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
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/optional.h"
#include "paddle/phi/common/int_array.h"
namespace paddle {
namespace prim {
using Tensor = paddle::experimental::Tensor;
using IntArray = paddle::experimental::IntArrayBase<paddle::experimental::Tensor>; 

template <typename T>
Tensor pow(const Tensor& x, const paddle::experimental::Scalar& y);

template <typename T>
Tensor scale(const Tensor& X,
             const paddle::experimental::Scalar& scale,
             float bias,
             bool bias_after_scale);

template <typename T>
Tensor multiply(const Tensor& x, const Tensor& y);

template <typename T>
Tensor expand(const Tensor& x, const IntArray& shape);

template <typename T>
Tensor unsqueeze(const Tensor& x, const IntArray& axis);
}  // namespace prim
}  // namespace paddle
