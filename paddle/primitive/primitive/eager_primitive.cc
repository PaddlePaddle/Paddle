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
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/primitive/primitive/primitive.h"

namespace paddle {
namespace primitive {
namespace experimental {

template <>
Tensor empty<Tensor>(const paddle::experimental::IntArray& shape,
                     phi::DataType dtype,
                     const paddle::Place& place) {
  if (dtype == phi::DataType::UNDEFINED) {
    dtype = phi::DataType::FLOAT32;
  }
  return empty_ad_func(shape, dtype, place);
}

template <>
Tensor empty_like<Tensor>(const paddle::Tensor& x,
                          phi::DataType dtype,
                          const paddle::Place& place) {
  if (dtype == phi::DataType::UNDEFINED) {
    dtype = phi::DataType::FLOAT32;
  }
  return empty_like_ad_func(x, dtype, place);
}

template <>
void set_output<Tensor>(const paddle::Tensor& x_tmp, paddle::Tensor* x) {
  x->set_impl(x_tmp.impl());
  x->set_autograd_meta(x_tmp.mutable_autograd_meta());
}

template <>
void by_pass<Tensor>(const paddle::Tensor& x, Tensor* out) {
  set_output<Tensor>(x, out);
}

template <>
Tensor tanh<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API tanh_ad_func call";
  return ::tanh_ad_func(x);
}
}  // namespace experimental
}  // namespace primitive
}  // namespace paddle
