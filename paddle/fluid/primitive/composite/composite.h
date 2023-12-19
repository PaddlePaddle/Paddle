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

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_tmp = cast<T>(x, phi::DataType::FLOAT32);
  }
  std::vector<int64_t> x_dim = common::vectorize<int64_t>(x_tmp.dims());
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
      sum_x / full<T>(common::vectorize(sum_x.dims()), value, sum_x.dtype());
  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

static bool valid_type(const DataType& dtype) {
  switch (dtype) {
    case phi::DataType::INT8:
    case phi::DataType::INT16:
    case phi::DataType::INT32:
    case phi::DataType::INT64:
    case phi::DataType::UINT8:
    case phi::DataType::UINT16:
    case phi::DataType::UINT32:
    case phi::DataType::UINT64:
    case phi::DataType::FLOAT16:
    case phi::DataType::FLOAT32:
    case phi::DataType::FLOAT64:
      return true;
    default:
      return false;
  }
}

template <typename T>
Tensor pow_decomp(const Tensor& x, const paddle::Scalar& y) {
  auto org_dtype = x.dtype();
  auto x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_cast = cast<T>(x, phi::DataType::FLOAT32);
  }

  Tensor y_full;
  if (valid_type(y.dtype())) {
    y_full = full<T>(common::vectorize(x_cast.dims()), y, x_cast.dtype());
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported data type: %s", phi::DataTypeToString(y.dtype())));
  }

  auto ans = elementwise_pow<T>(x_cast, y_full);
  if (need_cast) {
    return cast<T>(ans, org_dtype);
  } else {
    return ans;
  }
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm_decomp(
    const Tensor& x,
    const Tensor& run_mean,
    const Tensor& run_var,
    const paddle::optional<Tensor>& scale,
    const paddle::optional<Tensor>& bias,
    bool is_test,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool use_global_stats,
    bool trainable_statistics) {
  auto org_dtype = x.dtype();
  Tensor x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_cast = cast<T>(x, phi::DataType::FLOAT32);
  }

  std::vector<int64_t> x_dim = common::vectorize<int64_t>(x_cast.dims());
  int rank = x_dim.size();
  DataLayout data_layout_ = common::StringToDataLayout(data_layout);
  int feature_axis;
  if (data_layout_ == DataLayout::kNCHW) {
    feature_axis = 1;
  } else if (data_layout_ == DataLayout::kNHWC) {
    feature_axis = rank - 1;
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("Unknown storage order: %s", data_layout));
  }
  std::vector<int64_t> reduce_axes;
  for (int i = 0; i < rank; ++i) {
    if (i != feature_axis) {
      reduce_axes.push_back(i);
    }
  }
  std::vector<int64_t> stats_shape;
  for (int i = 0; i < rank; ++i) {
    if (find_value(reduce_axes, i) == false) {
      stats_shape.push_back(x_dim[i]);
    } else {
      stats_shape.push_back(1);
    }
  }

  Tensor half = full<T>(IntArray({1}), -0.5, x_cast.dtype());

  bool use_run_stat = (is_test && (!trainable_statistics)) || use_global_stats;
  Tensor x_hat;
  Tensor batch_mean;
  Tensor inv_std;
  Tensor run_mean_;
  Tensor run_var_;
  if (!use_run_stat) {
    batch_mean = mean_decomp<T>(x_cast, IntArray(reduce_axes), false);
    auto temp = mean_decomp<T>(x_cast * x_cast, IntArray(reduce_axes), false);
    auto batch_var = temp - batch_mean * batch_mean;
    inv_std = elementwise_pow<T>((batch_var + epsilon), half);
    if (data_layout_ == DataLayout::kNHWC) {
      x_hat = (x_cast - batch_mean) * inv_std;
    } else {
      x_hat = (x_cast - reshape<T>(batch_mean, stats_shape)) *
              reshape<T>(inv_std, stats_shape);
    }
    run_mean_ = run_mean * momentum + batch_mean * (1. - momentum);
    run_var_ = run_var * momentum + batch_var * (1. - momentum);
  } else {
    batch_mean =
        full<T>(common::vectorize(run_mean.dims()), 0, run_mean.dtype());
    auto batch_var =
        full<T>(common::vectorize(run_var.dims()), 0, run_var.dtype());
    inv_std = elementwise_pow<T>((batch_var + epsilon), half);
    if (data_layout_ == DataLayout::kNHWC) {
      x_hat =
          (x_cast - run_mean) * elementwise_pow<T>((run_var + epsilon), half);
    } else {
      x_hat = (x_cast - reshape<T>(run_mean, stats_shape)) *
              elementwise_pow<T>((reshape<T>(run_var, stats_shape) + epsilon),
                                 half);
    }
    run_mean_ = assign<T>(run_mean);
    run_var_ = assign<T>(run_var);
  }
  Tensor y;
  Tensor new_scale =
      scale ? scale.get()
            : full<T>(common::vectorize(x_cast.dims()), 1, x_cast.dtype());
  Tensor new_bias =
      bias ? bias.get()
           : full<T>(common::vectorize(x_cast.dims()), 0, x_cast.dtype());
  if (data_layout_ == DataLayout::kNHWC) {
    y = x_hat * new_scale + new_bias;
  } else {
    y = x_hat * reshape<T>(new_scale, stats_shape) +
        reshape<T>(new_bias, stats_shape);
  }
  if (need_cast) {
    y = cast<T>(y, org_dtype);
  }
  Tensor reserve_space;

  auto batch_mean_ = assign<T>(batch_mean);
  auto inv_std_ = assign<T>(inv_std);
  if (!use_run_stat) {
    return std::make_tuple(
        y, run_mean_, run_var_, batch_mean_, inv_std_, reserve_space);
  } else {
    return std::make_tuple(
        y, run_mean_, run_var_, batch_mean_, inv_std_, reserve_space);
  }
}

template <typename T>
Tensor softmax_decomp(const Tensor& x, const int& axis) {
  auto org_dtype = x.dtype();
  auto x_tmp = x;
  auto axis_tmp = IntArray({axis});

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_tmp = cast<T>(x, phi::DataType::FLOAT32);
  }

  auto max_tmp = max<T>(x_tmp, axis_tmp, true);
  auto molecular = exp<T>(x_tmp - max_tmp);
  auto res = molecular / sum<T>(molecular, axis_tmp, molecular.dtype(), true);

  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor stack_decomp(const std::vector<Tensor>& x, const int& axis) {
  std::vector<int64_t> axis_tmp = {axis};
  auto out_shape = get_expand_dims(x[0], axis_tmp);

  std::vector<Tensor> concat_x;
  for (size_t i = 0; i < x.size(); ++i) {
    concat_x.push_back(reshape<T>(x[i], out_shape));
  }
  return concat<T>(concat_x, axis);
}

template <typename T>
Tensor silu_decomp(const Tensor& x) {
  auto org_dtype = x.dtype();
  auto x_tmp = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_tmp = cast<T>(x, phi::DataType::FLOAT32);
  }

  // res = x / (1 + exp(-x))
  auto one = full<T>(common::vectorize(x.dims()), 1, x_tmp.dtype());
  auto exp_temp =
      exp<T>(full<T>(common::vectorize(x.dims()), -1, x_tmp.dtype()) * x_tmp);
  auto res = x_tmp / (exp_temp + one);
  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor relu_decomp(const Tensor& x) {
  return maximum<T>(x, full<T>(common::vectorize(x.dims()), 0.0, x.dtype()));
}

template <typename T>
Tensor rsqrt_decomp(const Tensor& x) {
  auto org_dtype = x.dtype();
  Tensor x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_cast = cast<T>(x, phi::DataType::FLOAT32);
  }

  auto ans = elementwise_pow<T>(
      x_cast, full<T>(common::vectorize(x_cast.dims()), -0.5, x_cast.dtype()));
  if (need_cast) {
    return cast<T>(ans, org_dtype);
  } else {
    return ans;
  }
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
std::tuple<Tensor, Tensor> unsqueeze_decomp(const Tensor& x,
                                            const IntArray& axis) {
  auto out_shape = get_expand_dims(x, axis.GetData());
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

  bool need_cast = is_half_dtype(org_dtype);

  // cast dtype to float32 if dtype =float16 or bfloat16
  if (need_cast) {
    x_cast = cast<T>(x_cast, phi::DataType::FLOAT32);
  }

  auto x_dim = common::vectorize<int64_t>(x.dims());
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
      full<T>(common::vectorize(var_tmp3.dims()), -0.5, var_tmp3.dtype()));
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
    } else {
      scale_cast = *scale_ptr;
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
    } else {
      bias_cast = *bias_ptr;
    }
    if (need_cast) {
      bias_cast = cast<T>(bias_cast, phi::DataType::FLOAT32);
    }
    out = out + bias_cast;
  }
  mean_ = reshape<T>(mean_, std::vector<int64_t>({-1}));
  variance = reshape<T>(variance, std::vector<int64_t>({-1}));

  // same as LayerNormInferMeta
  // x: float32 --> out: float32, mean: float32, variance: float32
  // x: float16 --> out: float16, mean: float32, variance: float32
  if (need_cast) {
    out = cast<T>(out, org_dtype);
  }

  return std::make_tuple(out, mean_, variance);
}

template <typename T>
Tensor full_like_decomp(const Tensor& x,
                        const paddle::Scalar& value,
                        const DataType& dtype,
                        const Place& place) {
  return full<T>(phi::vectorize(x.dims()), value, dtype, place);
}

template <typename T>
std::tuple<Tensor, Tensor> dropout_decomp(
    const Tensor& x,
    const paddle::optional<Tensor>& seed_tensor,
    const paddle::Scalar& p,
    bool is_test,
    const std::string& mode,
    const int seed,
    bool fix_seed) {
  auto org_dtype = x.dtype();
  bool upscale_in_train = false;
  if (mode == std::string("upscale_in_train")) {
    upscale_in_train = true;
  }

  int seed_tmp = 0;
  if (fix_seed) {
    seed_tmp = seed;
  }

  auto dtype_tmp = org_dtype;
  if (is_half_dtype(org_dtype)) {
    dtype_tmp = phi::DataType::FLOAT32;
  }

  auto uniform_tensor =
      uniform<T>(phi::vectorize(x.dims()), dtype_tmp, 0.0, 1.0, seed_tmp);
  auto mask =
      cast<T>(greater_equal<T>(uniform_tensor,
                               full<T>(phi::vectorize(x.dims()), p, dtype_tmp)),
              org_dtype);
  auto ones_p =
      full<T>(phi::vectorize(x.dims()), 1.0 - p.to<float>(), org_dtype);
  if (upscale_in_train) {
    if (is_test) {
      // inference: out = input
      return std::make_tuple(x, cast<T>(mask, phi::DataType::UINT8));
    } else {
      // train: out = input * mask / ( 1.0 - p )
      if (p.to<float>() == 1.0) {
        // Process p=1. for avoid devide zero error (x*mask/(1.0-p))
        auto zero = full<T>(phi::vectorize(x.dims()), 0.0, org_dtype);
        return std::make_tuple(x * zero, cast<T>(zero, phi::DataType::UINT8));
      } else {
        auto ans = (x * mask) / ones_p;
        return std::make_tuple(ans, cast<T>(mask, phi::DataType::UINT8));
      }
    }
  } else {
    if (is_test) {
      // inference: out = input * (1.0 - p)
      return std::make_tuple(x * ones_p, cast<T>(mask, phi::DataType::UINT8));
    } else {
      // train: out = input * mask
      return std::make_tuple(x * mask, cast<T>(mask, phi::DataType::UINT8));
    }
  }
}

template <typename T>
Tensor sqrt_decomp(const Tensor& x) {
  auto org_dtype = x.dtype();
  Tensor x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_cast = cast<T>(x, phi::DataType::FLOAT32);
  }

  auto ans = elementwise_pow<T>(
      x_cast, full<T>(common::vectorize(x_cast.dims()), 0.5, x_cast.dtype()));
  if (need_cast) {
    return cast<T>(ans, org_dtype);
  } else {
    return ans;
  }
}

template <typename T>
Tensor gelu_decomp(const Tensor& x, bool approximate) {
  const double PM_2_SQRTPI = 1.12837916709551257390; /* 2/sqrt(pi) */
  const double PM_SQRT1_2 = 0.70710678118654752440;  /* 1/sqrt(2) */

  auto org_dtype = x.dtype();
  auto half = full<T>(common::vectorize(x.dims()), 0.5, org_dtype);
  auto one = full<T>(common::vectorize(x.dims()), 1.0, org_dtype);
  if (approximate) {
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / \pi) * (x + 0.044715 * x^{3})))
    auto kAlpha = full<T>(
        common::vectorize(x.dims()), PM_2_SQRTPI * PM_SQRT1_2, org_dtype);
    auto GELU_CONSTANT =
        full<T>(common::vectorize(x.dims()), 0.044715, org_dtype);
    auto x_pow3 = elementwise_pow<T>(
        x, full<T>(common::vectorize(x.dims()), 3, org_dtype));
    auto tanh_out = tanh<T>(kAlpha * (x + x_pow3 * GELU_CONSTANT));

    auto res = x * half * (one + tanh_out);
    return res;
  } else {
    // gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))
    auto M_SQRT1_2T =
        full<T>(common::vectorize(x.dims()), PM_SQRT1_2, org_dtype);
    auto erf_out = one + erf<T>(x * M_SQRT1_2T);

    auto res = x * half * erf_out;
    return res;
  }
}

}  // namespace details

}  // namespace primitive
}  // namespace paddle
