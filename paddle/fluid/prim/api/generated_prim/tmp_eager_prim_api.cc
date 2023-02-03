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

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"

namespace paddle {
namespace prim {

template <>
Tensor divide<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API divide_ad_func call";
  return ::divide_ad_func(x, y);
}

template <>
Tensor expand<Tensor>(const Tensor& x, const IntArray& shape) {
  VLOG(4) << "Eager Prim API expand_ad_func call";
  return ::expand_ad_func(x, shape);
}

template <>
Tensor full<Tensor>(const IntArray& shape,
                    const Scalar& value,
                    DataType dtype,
                    const Place& place) {
  VLOG(4) << "Eager Prim API full_ad_func call";
  return ::full_ad_func(shape, value, dtype, place);
}

template <>
Tensor multiply<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API multiply_ad_func call";
  return ::multiply_ad_func(x, y);
}

template <>
Tensor pow<Tensor>(const Tensor& x, const Scalar& y) {
  VLOG(4) << "Eager Prim API pow_ad_func call";
  return ::pow_ad_func(x, y);
}

template <>
Tensor reshape<Tensor>(const Tensor& x, const IntArray& shape) {
  VLOG(4) << "Eager Prim API reshape_ad_func call";
  return ::reshape_ad_func(x, shape);
}

template <>
Tensor scale<Tensor>(const Tensor& x,
                     const Scalar& scale,
                     float bias,
                     bool bias_after_scale) {
  VLOG(4) << "Eager Prim API scale_ad_func call";
  return ::scale_ad_func(x, scale, bias, bias_after_scale);
}

template <>
Tensor sum<Tensor>(const Tensor& x,
                   const IntArray& axis,
                   DataType dtype,
                   bool keepdim) {
  VLOG(4) << "Eager Prim API sum_ad_func call";
  return ::sum_ad_func(x, axis, dtype, keepdim);
}

template <>
Tensor exp<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API exp_ad_func call";
  return ::exp_ad_func(x);
}

template <>
Tensor unsqueeze<Tensor>(const Tensor& x, const IntArray& axis) {
  VLOG(4) << "Eager Prim API unsqueeze_ad_func call";
  return ::unsqueeze_ad_func(x, axis);
}

}  // namespace prim
}  // namespace paddle
