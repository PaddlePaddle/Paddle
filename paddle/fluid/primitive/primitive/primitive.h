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
#include "paddle/fluid/primitive/backend/eager_backend.h"
#include "paddle/fluid/primitive/backend/static_backend.h"

namespace paddle {
namespace primitive {
// why exist this file?
// We provide this file to divide
// the primitive ops set in the backend.
// It will be called by the vjp composite
// rules and composite ops rules.
using Tensor = paddle::Tensor;
using IntArray = paddle::experimental::IntArray;

template <typename T>
Tensor divide(const Tensor& x, const Tensor& y) {
  return backend::divide<T>(x, y);
}

template <typename T>
Tensor add(const Tensor& x, const Tensor& y) {
  return backend::add<T>(x, y);
}

template <typename T>
Tensor multiply(const Tensor& x, const Tensor& y) {
  return backend::multiply<T>(x, y);
}

template <typename T>
Tensor elementwise_pow(const Tensor& x, const Tensor& y) {
  return backend::elementwise_pow<T>(x, y);
}

template <typename T>
Tensor scale(const Tensor& x,
             const Scalar& scale = 1.0,
             float bias = 0.0,
             bool bias_after_scale = true) {
  return backend::scale<T>(x, scale, bias, bias_after_scale);
}

template <typename T>
Tensor sum(const Tensor& x,
           const IntArray& axis = {},
           phi::DataType dtype = phi::DataType::UNDEFINED,
           bool keepdim = false) {
  return backend::sum<T>(x, axis, dtype, keepdim);
}

template <typename T>
Tensor full(const IntArray& shape,
            const Scalar& value,
            phi::DataType dtype = phi::DataType::FLOAT32,
            phi::Place place = phi::CPUPlace()) {
  return backend::full<T>(shape, value, dtype, place);
}

template <typename T>
std::tuple<Tensor, Tensor> reshape(const Tensor& x, const IntArray& shape) {
  return backend::reshape<T>(x, shape);
}

template <typename T>
Tensor expand(const Tensor& x, const IntArray& shape) {
  return backend::expand<T>(x, shape);
}

template <typename T>
Tensor tile(const Tensor& x, const IntArray& repeat_times = {}) {
  return backend::tile<T>(x, repeat_times);
}
}  // namespace primitive
}  // namespace paddle
