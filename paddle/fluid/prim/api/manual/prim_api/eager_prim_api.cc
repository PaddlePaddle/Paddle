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

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/prim/api/manual/prim_api/prim_api.h"
#include "paddle/phi/capi/include/wrapper_base.h"
namespace paddle {
namespace prim {
template <>
Tensor pow<Tensor>(const Tensor& x, const paddle::experimental::Scalar& y) {
  return ::pow_ad_func(x, y);
}

template <>
Tensor scale<Tensor>(const Tensor& x,
                     const paddle::experimental::Scalar& scale,
                     float bias,
                     bool bias_after_scale) {
  return ::scale_ad_func(x, scale, bias, bias_after_scale);
}

template <>
Tensor multiply<Tensor>(const Tensor& x, const Tensor& y) {
  return ::multiply_ad_func(x, y);
}

template <>
Tensor expand<Tensor>(const Tensor& x, const IntArray& shape) {
  return ::expand_ad_func(x, shape);
}

template <>
Tensor unsqueeze<Tensor>(const Tensor& x, const IntArray& axis) {
  return ::unsqueeze_ad_func(x, axis);
}

template <>
Tensor divide<Tensor>(const Tensor& x, const Tensor& y) {
  return ::divide_ad_func(x, y);
}

template <>
Tensor full<Tensor>(paddle::experimental::IntArray shape,
                    paddle::experimental::Scalar value,
                    paddle::experimental::DataType dtype,
                    paddle::platform::Place place) {
  return ::full_ad_func(shape, value, dtype, place);
}
template <>
Tensor sum<Tensor>(Tensor x, IntArray axis, DataType dtype, bool keepdim) {
  return ::sum_ad_func(x, axis, dtype, keepdim);
}

template <>
Tensor reshape<Tensor>(Tensor x, IntArray shape) {
  return ::reshape_ad_func(x, shape);
}

template <>
Tensor exp<Tensor>(const Tensor& x) {
  return ::exp_ad_func(x);
}

template <typename T>
Tensor expand(const Tensor& x, const IntArray& shape) {
  return ::expand_ad_func(x, shape);
}
}  // namespace prim
}  // namespace paddle
