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
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"

namespace paddle {
namespace primitive {
namespace details {

template <typename T>
Tensor mean_decomp(const Tensor& x, const IntArray& axis, bool keepdim) {
  auto org_dtype = x.dtype();
  auto x_tmp = x;
  bool need_cast = org_dtype == phi::DataType::FLOAT16 ||
                   org_dtype == phi::DataType::BFLOAT16;
  if (need_cast) {
    x_tmp = cast<T>(x, phi::DataType::FLOAT32);
  }
  std::vector<int64_t> x_dim = phi::vectorize<int64_t>(x_tmp.dims());
  int64_t axis_size = axis.size();
  int64_t x_dim_size = x_dim.size();
  auto axis_ = std::vector<int64_t>();
  if (axis_size == 0) {
    for (int64_t i = 0; i < x_dim_size; i++) {
      axis_.push_back(i);
    }
  } else {
    axis_ = axis.GetData();
    for (int64_t i = 0; i < axis_size; i++) {
      if (axis[i] < 0) {
        axis_[i] = axis[i] + x_dim_size;
      }
    }
  }

  int64_t value = 1;
  for (size_t i = 0; i < axis_.size(); i++) {
    value *= x_dim[axis_[i]];
  }
  auto sum_x = sum<T>(x_tmp, IntArray(axis_), x_tmp.dtype(), keepdim);
  auto res =
      sum_x / full<T>(phi::vectorize(sum_x.dims()), value, sum_x.dtype());
  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor softmax_decomp(const Tensor& x, const int& axis) {
  auto org_dtype = x.dtype();
  auto x_tmp = x;
  auto axis_tmp = IntArray({axis});

  bool need_cast =
      org_dtype == phi::DataType::FLOAT16 || org_dtype == phi::DataType::UINT16;
  if (need_cast) {
    x_tmp = cast<T>(x, phi::DataType::FLOAT32);
  }

  auto max_tmp = max<T>(x_tmp, axis_tmp, true);
  auto molecular = exp<T>(subtract<T>(x_tmp, max_tmp));

  auto denominator = sum<T>(molecular, axis_tmp, molecular.dtype(), true);
  auto res = divide<T>(molecular, denominator);

  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor relu_decomp(const Tensor& x) {
  return maximum<T>(x, full<T>(phi::vectorize(x.dims()), 0.0, x.dtype()));
}

template <typename T>
std::tuple<Tensor, Tensor> squeeze_decomp(const Tensor& x,
                                          const IntArray& axis) {
  auto axis_ = process_dims(x, axis.GetData());
  auto out_shape = get_squeeze_dims(x, axis_);
  Tensor out = reshape<T>(x, out_shape);
  Tensor xshape;
  return std::make_tuple(out, xshape);
}

template <typename T>
Tensor add_n_decomp(const std::vector<Tensor>& x) {
  Tensor res = x[0];
  for (size_t i = 1; i < x.size(); i++) {
    res = res + x[i];
  }
  return res;
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> layer_norm_decomp(
    const Tensor& x,
    const paddle::optional<Tensor>& scale,
    const paddle::optional<Tensor>& bias,
    float epsilon,
    int begin_norm_axis) {
  std::vector<int64_t> axis;
  auto org_dtype = x.dtype();
  Tensor x_cast = x;

  bool need_cast = org_dtype == phi::DataType::FLOAT16 ||
                   org_dtype == phi::DataType::BFLOAT16;

  // cast dtype to float32 if dtype =float16 or bfloat16
  if (need_cast) {
    x_cast = cast<T>(x_cast, phi::DataType::FLOAT32);
  }

  auto x_dim = phi::vectorize<int64_t>(x.dims());
  for (size_t i = begin_norm_axis; i < x_dim.size(); i++) {
    axis.push_back(static_cast<int64_t>(i));
  }
  auto mean_ = mean_decomp<T>(x_cast, IntArray(axis), true);
  auto difference = x_cast - mean_;
  auto var_tmp1 = difference * difference;
  auto variance = mean_decomp<T>(var_tmp1, IntArray(axis), true);
  auto var_tmp3 = variance + epsilon;
  auto rsqrt_var = elementwise_pow<T>(
      var_tmp3,
      full<T>(phi::vectorize(var_tmp3.dims()), -0.5, var_tmp3.dtype()));
  auto out = difference * rsqrt_var;

  auto scale_ptr = scale.get_ptr();
  auto bias_ptr = bias.get_ptr();

  std::vector<int64_t> slice_shape;
  for (int64_t i = begin_norm_axis; i < static_cast<int64_t>(x_dim.size());
       i++) {
    slice_shape.push_back(x_dim[i]);
  }
  Tensor scale_cast;
  if (scale_ptr) {
    if (slice_shape != scale_ptr->shape()) {
      scale_cast = reshape<T>(*scale_ptr, slice_shape);
    }
    if (need_cast) {
      scale_cast = cast<T>(scale_cast, phi::DataType::FLOAT32);
    }
    out = out * scale_cast;
  }
  Tensor bias_cast;
  if (bias_ptr) {
    if (slice_shape != bias_ptr->shape()) {
      bias_cast = reshape<T>(*bias_ptr, slice_shape);
    }
    if (need_cast) {
      bias_cast = cast<T>(bias_cast, phi::DataType::FLOAT32);
    }
    out = out + bias_cast;
  }
  mean_ = reshape<T>(mean_, std::vector<int64_t>({-1}));
  variance = reshape<T>(variance, std::vector<int64_t>({-1}));

  if (need_cast) {
    out = cast<T>(out, org_dtype);
    mean_ = cast<T>(mean_, org_dtype);
    variance = cast<T>(variance, org_dtype);
  }

  return std::make_tuple(out, mean_, variance);
}

}  // namespace details

}  // namespace primitive
}  // namespace paddle
