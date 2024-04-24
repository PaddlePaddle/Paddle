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

#include <numeric>
#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"

namespace paddle {
namespace primitive {
namespace details {

// empty_shape means x.shape=[]
static std::vector<int64_t> empty_shape;

template <typename T>
static Tensor get_slice(const Tensor& x, int64_t idx) {
  return slice<T>(x, {0}, {idx}, {idx + 1}, {1}, {});
}

template <typename T>
static Tensor get_slice_vec(const Tensor& x,
                            int64_t start_idx,
                            int64_t end_idx) {
  return slice<T>(x, {0}, {start_idx}, {end_idx}, {1}, {});
}

template <typename T>
Tensor any_decomp(const Tensor& x, const IntArray& axis, bool keepdim) {
  auto org_dtype = x.dtype();

  auto res = cast<T>(sum<T>(x, axis, org_dtype, keepdim), DataType::BOOL);
  if (org_dtype != DataType::BOOL) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor mean_decomp(const Tensor& x, const IntArray& axis, bool keepdim) {
  auto org_dtype = x.dtype();
  auto x_tmp = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_tmp = cast<T>(x, DataType::FLOAT32);
  }
  std::vector<int64_t> x_dim = x_tmp.shape();
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
  auto sum_x = sum<T>(x_tmp, axis_, x_tmp.dtype(), keepdim);

  Tensor value;
  bool switch_dynamic = false;
  for (const int64_t& idx : axis_) {
    if (x_dim[idx] == -1) {
      switch_dynamic = true;
    }
  }
  if (switch_dynamic) {
    // dynamic shape branch
    std::vector<int64_t> gather_idx = {int64_t(axis_.size()), 1};
    Tensor idx =
        reshape<T>(full_int_array<T>(axis_, DataType::INT64), gather_idx);
    auto axis_dims = cast<T>(gather_nd<T>(shape<T>(x), idx), sum_x.dtype());
    value = prod<T>(axis_dims, {0}, false, false);
  } else {
    int64_t value_ = 1;
    for (size_t i = 0; i < axis_.size(); i++) {
      value_ *= x_dim[axis_[i]];
    }
    value = full<T>(empty_shape, value_, sum_x.dtype());
  }

  Tensor res = sum_x / value;

  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

static bool valid_type(const DataType& dtype) {
  switch (dtype) {
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
    case DataType::INT64:
    case DataType::UINT8:
    case DataType::UINT16:
    case DataType::UINT32:
    case DataType::UINT64:
    case DataType::FLOAT16:
    case DataType::FLOAT32:
    case DataType::FLOAT64:
      return true;
    default:
      return false;
  }
}

template <typename T>
Tensor p_norm_decomp(const Tensor& x,
                     const float& porder = 2.0,
                     const int& axis = -1,
                     const float epsilon = 1.0e-12f,
                     const bool& keepdim = false,
                     const bool& asvector = false) {
  auto org_dtype = x.dtype();
  auto x_tmp = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_tmp = cast<T>(x, DataType::FLOAT32);
  }

  Tensor res;
  if (porder == 0.0) {
    // 0-norm
    auto zero = full<T>(empty_shape, 0, x_tmp.dtype());
    auto none_zero = not_equal<T>(x_tmp, zero);
    res = cast<T>(none_zero, x_tmp.dtype());
    res = sum<T>(res, {axis}, x_tmp.dtype(), keepdim);
  } else if (porder == 1.0) {
    // 1-norm
    res = abs<T>(x_tmp);
    res = sum<T>(res, {axis}, x_tmp.dtype(), keepdim);
  } else if (porder == 2.0) {
    // 2-norm
    res = sqrt<T>(sum<T>(x_tmp * x_tmp, {axis}, x_tmp.dtype(), keepdim));
  } else if (porder == INFINITY) {
    // +INF-norm
    res = abs<T>(x_tmp);
    res = max<T>(x_tmp, {axis}, keepdim);
  } else if (porder == -INFINITY) {
    // -INF-norm
    res = abs<T>(x_tmp);
    res = min<T>(x_tmp, {axis}, keepdim);
  } else {
    // vanilla p-norm
    auto porder_tensor = full<T>(empty_shape, porder, x_tmp.dtype());
    auto inv_porder_tensor = full<T>(empty_shape, 1 / porder, x_tmp.dtype());
    res = elementwise_pow<T>(x_tmp, porder_tensor);
    res = sum<T>(res, {axis}, x_tmp.dtype(), keepdim);
    res = elementwise_pow<T>(res, inv_porder_tensor);
  }

  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor pow_decomp(const Tensor& x, const paddle::Scalar& y) {
  auto org_dtype = x.dtype();
  auto x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_cast = cast<T>(x, DataType::FLOAT32);
  }

  Tensor y_full;
  if (valid_type(y.dtype())) {
    y_full = full<T>(empty_shape, y, x_cast.dtype());
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
Tensor reciprocal_decomp(const Tensor& x) {
  return full<T>(empty_shape, 1.0, x.dtype()) / x;
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
    x_cast = cast<T>(x, DataType::FLOAT32);
  }

  std::vector<int64_t> x_dim = x_cast.shape();
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

  Tensor half = full<T>(empty_shape, -0.5, x_cast.dtype());

  bool use_run_stat = (is_test && (!trainable_statistics)) || use_global_stats;
  Tensor x_hat;
  Tensor batch_mean;
  Tensor inv_std;
  Tensor run_mean_;
  Tensor run_var_;
  if (!use_run_stat) {
    batch_mean = mean_decomp<T>(x_cast, reduce_axes, false);
    auto temp = mean_decomp<T>(x_cast * x_cast, reduce_axes, false);
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
    assign_out_<T>(run_mean_, run_mean);
    assign_out_<T>(run_var_, run_var);
  } else {
    batch_mean = full<T>(run_mean.shape(), 0, run_mean.dtype());
    auto batch_var = full<T>(run_var.shape(), 0, run_var.dtype());
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
      scale ? scale.get() : full<T>(empty_shape, 1, x_cast.dtype());
  Tensor new_bias = bias ? bias.get() : full<T>(empty_shape, 0, x_cast.dtype());
  if (data_layout_ == DataLayout::kNHWC) {
    y = x_hat * new_scale + new_bias;
  } else {
    y = x_hat * reshape<T>(new_scale, stats_shape) +
        reshape<T>(new_bias, stats_shape);
  }
  Tensor reserve_space;

  auto batch_mean_ = assign<T>(batch_mean);
  auto inv_std_ = assign<T>(inv_std);
  if (need_cast) {
    y = cast<T>(y, org_dtype);
  }
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

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_tmp = cast<T>(x, DataType::FLOAT32);
  }

  auto max_tmp = max<T>(x_tmp, {axis}, true);
  auto molecular = exp<T>(x_tmp - max_tmp);
  auto res = molecular / sum<T>(molecular, {axis}, molecular.dtype(), true);

  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor log_softmax_decomp(const Tensor& x, const int& axis) {
  auto org_dtype = x.dtype();
  auto x_tmp = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_tmp = cast<T>(x, DataType::FLOAT32);
  }

  auto max_tmp = max<T>(x_tmp, {axis}, true);
  auto sub = x_tmp - max_tmp;
  auto molecular = exp<T>(sub);
  auto res = sub - log<T>(sum<T>(molecular, {axis}, molecular.dtype(), true));

  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor stack_decomp(const std::vector<Tensor>& x, const int& axis) {
  std::vector<Tensor> concat_x;
  if (has_dynamic_shape(x[0].shape())) {
    Tensor out_shape = shape<T>(unsqueeze<T>(x[0], {axis}));
    for (size_t i = 0; i < x.size(); ++i) {
      concat_x.push_back(backend::reshape<T>(x[i], out_shape));
    }
  } else {
    std::vector<int64_t> axis_tmp = {axis};
    std::vector<int64_t> out_shape = get_expand_dims(x[0], axis_tmp);
    for (size_t i = 0; i < x.size(); ++i) {
      concat_x.push_back(reshape<T>(x[i], out_shape));
    }
  }

  return concat<T>(concat_x, axis);
}

template <typename T>
Tensor silu_decomp(const Tensor& x) {
  auto org_dtype = x.dtype();
  auto x_tmp = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_tmp = cast<T>(x, DataType::FLOAT32);
  }
  auto res = x_tmp * sigmoid<T>(x_tmp);
  if (need_cast) {
    return cast<T>(res, org_dtype);
  } else {
    return res;
  }
}

template <typename T>
Tensor swiglu_decomp(const Tensor& x, const paddle::optional<Tensor>& y) {
  if (y) {
    return silu_decomp<T>(x) * y.get();
  } else {
    int axis = x.shape().size() - 1;
    int num = 2;
    std::vector<Tensor> xs = backend::split_with_num<T>(x, num, axis);
    return silu_decomp<T>(xs[0]) * xs[1];
  }
}

template <typename T>
Tensor relu_decomp(const Tensor& x) {
  return maximum<T>(x, full<T>(empty_shape, 0.0, x.dtype()));
}

template <typename T>
Tensor relu6_decomp(const Tensor& x) {
  auto tmp = maximum<T>(x, full<T>(empty_shape, 0.0, x.dtype()));
  auto res = minimum<T>(tmp, full<T>(empty_shape, 6.0, x.dtype()));
  return res;
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
  if (has_dynamic_shape(x.shape())) {
    std::vector<int64_t> axis;
    auto org_dtype = x.dtype();
    Tensor x_cast = x;

    bool need_cast = is_half_dtype(org_dtype);

    // cast dtype to float32 if dtype =float16 or bfloat16
    if (need_cast) {
      x_cast = cast<T>(x_cast, DataType::FLOAT32);
    }

    auto x_dim = x.shape();
    for (size_t i = begin_norm_axis; i < x_dim.size(); i++) {
      axis.push_back(static_cast<int64_t>(i));
    }
    auto mean_ = mean_decomp<T>(x_cast, axis, true);
    auto difference = x_cast - mean_;
    auto var_tmp1 = difference * difference;
    auto variance = mean_decomp<T>(var_tmp1, axis, true);
    auto var_tmp3 = variance + full<T>(empty_shape, epsilon, variance.dtype());
    auto rsqrt_var = rsqrt<T>(var_tmp3);
    auto out = difference * rsqrt_var;

    Tensor slice_shape_l = get_slice_vec<T>(shape<T>(x), 0, begin_norm_axis);
    Tensor slice_shape_r =
        get_slice_vec<T>(shape<T>(x), begin_norm_axis, x_dim.size());
    Tensor scale_cast;
    if (scale) {
      scale_cast = backend::reshape_with_tensor<T>(scale.get(), slice_shape_r);
      if (need_cast) {
        scale_cast = cast<T>(scale_cast, DataType::FLOAT32);
      }
      out = out * scale_cast;
    }
    Tensor bias_cast;
    if (bias) {
      bias_cast = backend::reshape_with_tensor<T>(bias.get(), slice_shape_r);
      if (need_cast) {
        bias_cast = cast<T>(bias_cast, DataType::FLOAT32);
      }
      out = out + bias_cast;
    }
    mean_ = backend::reshape_with_tensor<T>(mean_, slice_shape_l);
    variance = backend::reshape_with_tensor<T>(variance, slice_shape_l);

    // same as LayerNormInferMeta
    // x: float32 --> out: float32, mean: float32, variance: float32
    // x: float16 --> out: float16, mean: float32, variance: float32
    if (need_cast) {
      out = cast<T>(out, org_dtype);
    }

    return std::make_tuple(out, mean_, variance);
  }

  std::vector<int64_t> axis;
  auto org_dtype = x.dtype();
  Tensor x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);

  // cast dtype to float32 if dtype =float16 or bfloat16
  if (need_cast) {
    x_cast = cast<T>(x_cast, DataType::FLOAT32);
  }

  auto x_dim = x.shape();
  for (size_t i = begin_norm_axis; i < x_dim.size(); i++) {
    axis.push_back(static_cast<int64_t>(i));
  }
  auto mean_ = mean_decomp<T>(x_cast, axis, true);
  auto difference = x_cast - mean_;
  auto var_tmp1 = difference * difference;
  auto variance = mean_decomp<T>(var_tmp1, axis, true);
  auto var_tmp3 = variance + epsilon;
  auto rsqrt_var = rsqrt<T>(var_tmp3);
  auto out = difference * rsqrt_var;

  std::vector<int64_t> slice_shape_l;
  std::vector<int64_t> slice_shape_r;
  for (int64_t i = 0; i < static_cast<int64_t>(x_dim.size()); i++) {
    if (i < begin_norm_axis) {
      slice_shape_l.push_back(x_dim[i]);
    } else {
      slice_shape_r.push_back(x_dim[i]);
    }
  }
  Tensor scale_cast;
  if (scale) {
    scale_cast = reshape<T>(scale.get(), slice_shape_r);
    if (need_cast) {
      scale_cast = cast<T>(scale_cast, DataType::FLOAT32);
    }
    out = out * scale_cast;
  }
  Tensor bias_cast;
  if (bias) {
    bias_cast = reshape<T>(bias.get(), slice_shape_r);
    if (need_cast) {
      bias_cast = cast<T>(bias_cast, DataType::FLOAT32);
    }
    out = out + bias_cast;
  }
  mean_ = reshape<T>(mean_, slice_shape_l);
  variance = reshape<T>(variance, slice_shape_l);

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
  std::vector<int64_t> x_shape = x.shape();
  if (has_dynamic_shape(x_shape)) {
    return backend::full_with_tensor<T>(shape<T>(x), value, x.dtype());
  } else {
    return full<T>(x_shape, value, dtype, place);
  }
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
    dtype_tmp = DataType::FLOAT32;
  }

  auto uniform_tensor =
      uniform<T>(phi::vectorize(x.dims()), dtype_tmp, 0.0, 1.0, seed_tmp);
  auto mask = cast<T>(
      greater_equal<T>(uniform_tensor, full<T>(empty_shape, p, dtype_tmp)),
      org_dtype);
  auto ones_p = full<T>(empty_shape, 1.0 - p.to<float>(), org_dtype);
  if (upscale_in_train) {
    if (is_test) {
      // inference: out = input
      return std::make_tuple(x, cast<T>(mask, DataType::UINT8));
    } else {
      // train: out = input * mask / ( 1.0 - p )
      if (p.to<float>() == 1.0) {
        // Process p=1. for avoid divide zero error (x*mask/(1.0-p))
        auto zero = full<T>(empty_shape, 0.0, org_dtype);
        return std::make_tuple(x * zero, cast<T>(zero, DataType::UINT8));
      } else {
        auto ans = (x * mask) / ones_p;
        return std::make_tuple(ans, cast<T>(mask, DataType::UINT8));
      }
    }
  } else {
    if (is_test) {
      // inference: out = input * (1.0 - p)
      return std::make_tuple(x * ones_p, cast<T>(mask, DataType::UINT8));
    } else {
      // train: out = input * mask
      return std::make_tuple(x * mask, cast<T>(mask, DataType::UINT8));
    }
  }
}

template <typename T>
Tensor gelu_decomp(const Tensor& x, bool approximate) {
  const double PM_2_SQRTPI = 1.12837916709551257390; /* 2/sqrt(pi) */
  const double PM_SQRT1_2 = 0.70710678118654752440;  /* 1/sqrt(2) */

  auto org_dtype = x.dtype();
  auto half = full<T>(empty_shape, 0.5, org_dtype);
  auto one = full<T>(empty_shape, 1.0, org_dtype);
  if (approximate) {
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / \pi) * (x + 0.044715 * x^{3})))
    auto kAlpha = full<T>(empty_shape, PM_2_SQRTPI * PM_SQRT1_2, org_dtype);
    auto GELU_CONSTANT = full<T>(empty_shape, 0.044715, org_dtype);
    auto x_pow3 = elementwise_pow<T>(x, full<T>(empty_shape, 3, org_dtype));
    auto tanh_out = tanh<T>(kAlpha * (x + x_pow3 * GELU_CONSTANT));

    auto res = x * half * (one + tanh_out);
    return res;
  } else {
    // gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))
    auto M_SQRT1_2T = full<T>(empty_shape, PM_SQRT1_2, org_dtype);
    auto erf_out = one + erf<T>(x * M_SQRT1_2T);

    auto res = x * half * erf_out;
    return res;
  }
}

template <typename T>
Tensor hardswish_decomp(const Tensor& x) {
  const double OFFSET = 3.0;
  const double THRESHOLD = 6.0;
  const double SCALE = 6.0;

  // out = minimum(maximum(x + offset, 0), threshold) * x / scale
  auto minimum_out =
      minimum<T>(maximum<T>(x + full<T>(empty_shape, OFFSET, x.dtype()),
                            full<T>(empty_shape, 0.0, x.dtype())),
                 full<T>(empty_shape, THRESHOLD, x.dtype()));
  return (minimum_out * x) / full<T>(empty_shape, SCALE, x.dtype());
}

template <typename T>
Tensor leaky_relu_decomp(const Tensor& x, float negative_slope) {
  auto multiply_tmp = full<T>(empty_shape, negative_slope, x.dtype()) * x;
  if (negative_slope < 1.0) {
    return maximum<T>(x, multiply_tmp);
  } else {
    return minimum<T>(x, multiply_tmp);
  }
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> instance_norm_decomp(
    const Tensor& x,
    const paddle::optional<Tensor>& scale,
    const paddle::optional<Tensor>& bias,
    float epsilon) {
  auto org_dtype = x.dtype();
  Tensor x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_cast = cast<T>(x, DataType::FLOAT32);
  }

  std::vector<int64_t> axis;
  auto x_dim = x.shape();
  for (size_t i = 2; i < x_dim.size(); i++) {
    axis.push_back(static_cast<int64_t>(i));
  }

  // out = (x - mean(x)) / sqrt(var + epsilon))
  // var = mean((x-mean(x))^2)
  auto mean_ = mean_decomp<T>(x_cast, axis, true);
  auto difference = x_cast - mean_;
  auto var_tmp1 = difference * difference;
  auto variance = mean_decomp<T>(var_tmp1, axis, true);
  auto var_tmp3 = variance + epsilon;
  auto rsqrt_var = rsqrt<T>(var_tmp3);
  auto out = difference * rsqrt_var;

  std::vector<int64_t> slice_shape(x_dim.size(), 1);
  slice_shape[1] = x_dim[1];

  Tensor scale_cast;
  if (scale) {
    scale_cast = reshape<T>(scale.get(), slice_shape);
    if (need_cast) {
      scale_cast = cast<T>(scale_cast, DataType::FLOAT32);
    }
    out = out * scale_cast;
  }
  Tensor bias_cast;
  if (bias) {
    bias_cast = reshape<T>(bias.get(), slice_shape);
    if (need_cast) {
      bias_cast = cast<T>(bias_cast, DataType::FLOAT32);
    }
    out = out + bias_cast;
  }

  std::vector<int64_t> res_shape(1, -1);
  auto mean_out = reshape<T>(mean_, res_shape);
  auto variance_out = reshape<T>(rsqrt_var, res_shape);

  Tensor res;
  if (need_cast) {
    res = cast<T>(out, org_dtype);
  } else {
    res = out;
  }

  return std::make_tuple(res, mean_out, variance_out);
}

template <typename T>
std::tuple<Tensor, Tensor> flatten_decomp(const Tensor& x,
                                          int start_axis,
                                          int end_axis) {
  auto x_dim = x.shape();
  if (x_dim.size() == 0) {
    start_axis = 0;
    end_axis = 0;
  }
  if (end_axis < start_axis) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "end_axis must be greater than or equal to start_axis."));
  }

  if (has_dynamic_shape(x.shape())) {
    auto x_shape = shape<T>(x);
    Tensor x_shape_tensor = full<T>({1}, 0, x_shape.dtype());
    std::vector<Tensor> tmp_shape;
    tmp_shape.push_back(x_shape_tensor);
    for (size_t i = 0; i < x_dim.size(); i++) {
      tmp_shape.push_back(get_slice<T>(x_shape, i));
    }
    x_shape_tensor = concat<T>(tmp_shape);
    x_shape_tensor =
        backend::full_with_tensor<T>(x_shape_tensor, 0.0, DataType::FLOAT32);
    if (end_axis == start_axis) {
      return std::make_tuple(backend::reshape<T>(x, x_shape), x_shape_tensor);
    }
    std::vector<Tensor> out_shape;

    for (size_t i = 0; i < x_dim.size();) {
      if (i == static_cast<size_t>(start_axis)) {
        Tensor flat = get_slice<T>(x_shape, i);

        for (auto t = start_axis + 1; t <= end_axis; ++t) {
          flat = flat * get_slice<T>(x_shape, t);
        }
        out_shape.push_back(flat);
        i = end_axis + 1;
      } else {
        out_shape.push_back(get_slice<T>(x_shape, i));
        i++;
      }
    }

    Tensor out_shape_tensor = concat<T>(out_shape);
    return std::make_tuple(backend::reshape<T>(x, out_shape_tensor),
                           x_shape_tensor);
  } else {
    std::vector<int64_t> tmp_shape(x_dim);
    tmp_shape.insert(tmp_shape.begin(), 0);
    auto xshape = full<T>(tmp_shape, 0.0, DataType::FLOAT32);
    if (x_dim.size() == 0) {
      std::vector<int64_t> res_shape(1, 1);
      return std::make_tuple(reshape<T>(x, res_shape), xshape);
    }
    if (end_axis == start_axis) {
      return std::make_tuple(reshape<T>(x, x_dim), xshape);
    }

    int slice_numel = 1;
    for (int i = start_axis; i <= end_axis; ++i) {
      slice_numel *= x_dim[i];
    }
    std::vector<int64_t> out_shape;
    for (int i = 0; i < start_axis; ++i) {
      out_shape.push_back(x_dim[i]);
    }
    out_shape.push_back(slice_numel);
    for (size_t i = end_axis + 1; i < x_dim.size(); ++i) {
      out_shape.push_back(x_dim[i]);
    }

    return std::make_tuple(reshape<T>(x, out_shape), xshape);
  }
}

template <typename T>
Tensor index_select_decomp(const Tensor& x, const Tensor& index, int axis) {
  int axis_tmp = axis;
  if (axis < 0) {
    axis_tmp += x.dims().size();
  }

  return gather<T>(x, index, axis_tmp);
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> group_norm_decomp(
    const Tensor& x,
    const paddle::optional<Tensor>& scale,
    const paddle::optional<Tensor>& bias,
    const float epsilon,
    const int groups,
    const std::string& data_format) {
  std::vector<int64_t> c_axis;
  if (data_format == "NCHW") {
    c_axis = {1};
  } else if (data_format == "NHWC") {
    c_axis = {1, 3};
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Only support NCHW and NHWC format."));
  }
  size_t rank = x.shape().size();
  if (rank < 3 || rank > 5) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support NCHW and NHWC format in rank {3, 4, 5}."));
  }

  auto org_dtype = x.dtype();
  Tensor x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_cast = cast<T>(x, DataType::FLOAT32);
  }

  Tensor x_dim_t;
  Tensor out, mean_, var_;
  if (has_dynamic_shape(x_cast.shape())) {
    x_dim_t = shape<T>(x_cast);
    Tensor tar_shape;
    if (data_format == "NCHW") {
      tar_shape = get_slice<T>(x_dim_t, 0) * groups;
      Tensor dim_1 = full<T>({1}, -1, x_dim_t.type());
      tar_shape = concat<T>({tar_shape, dim_1});
    } else {
      Tensor N_shape = get_slice<T>(x_dim_t, 0);
      Tensor dim_1 = full<T>({1}, -1, x_dim_t.type());
      Tensor C_shape = get_slice<T>(x_dim_t, rank - 1);
      Tensor dim_g = full<T>({1}, groups, x_dim_t.type());
      Tensor dim_c_div_g = cast<T>(C_shape / dim_g, x_dim_t.type());
      tar_shape = concat<T>({N_shape, dim_1, dim_g, dim_c_div_g});
    }
    x_cast = backend::reshape<T>(x_cast, tar_shape);
    mean_ = mean_decomp<T>(x_cast, c_axis, true);
    Tensor var_tmp_ =
        mean_decomp<T>(x_cast * x_cast, c_axis, true) - mean_ * mean_;
    var_ = maximum<T>(
        var_tmp_,
        backend::full_with_tensor<T>(shape<T>(var_tmp_), 0, var_tmp_.dtype()));
    Tensor var_inv =
        rsqrt<T>(var_ + full<T>(empty_shape, epsilon, var_.dtype()));
    Tensor res = (x_cast - mean_) * var_inv;
    out = backend::reshape<T>(res, x_dim_t);
  } else {
    auto x_dim = x_cast.shape();
    if (data_format == "NCHW") {
      x_cast = reshape<T>(x_cast, {x_dim[0] * groups, -1});
    } else {
      int c_div_g = x_dim[rank - 1] / groups;
      x_cast = reshape<T>(x_cast, {x_dim[0], -1, groups, c_div_g});
    }
    mean_ = mean_decomp<T>(x_cast, c_axis, true);
    auto var_tmp_ =
        mean_decomp<T>(x_cast * x_cast, c_axis, true) - mean_ * mean_;
    var_ = maximum<T>(var_tmp_, full<T>(var_tmp_.shape(), 0, var_tmp_.dtype()));
    auto var_inv = rsqrt<T>(var_ + full<T>(empty_shape, epsilon, var_.dtype()));
    auto res = (x_cast - mean_) * var_inv;
    out = reshape<T>(res, x_dim);
  }

  std::vector<int64_t> slice_bias_shape;
  slice_bias_shape = {-1};
  for (size_t i = 0; i < rank - 2; i++) {
    slice_bias_shape.push_back(1);
  }
  Tensor scale_cast;
  if (scale) {
    if (data_format == "NCHW") {
      scale_cast = reshape<T>(scale.get(), slice_bias_shape);
    } else {
      scale_cast = scale.get();
    }
    if (need_cast) {
      scale_cast = cast<T>(scale_cast, DataType::FLOAT32);
    }
    out = out * scale_cast;
  }
  Tensor bias_cast;
  if (bias) {
    if (data_format == "NCHW") {
      bias_cast = reshape<T>(bias.get(), slice_bias_shape);
    } else {
      bias_cast = bias.get();
    }
    if (need_cast) {
      bias_cast = cast<T>(bias_cast, DataType::FLOAT32);
    }
    out = out + bias_cast;
  }
  Tensor mean_out, var_out;
  if (has_dynamic_shape(x_cast.shape())) {
    Tensor x_shape = get_slice<T>(x_dim_t, 0);
    Tensor dim_1 = full<T>({1}, groups, x_shape.type());
    x_shape = concat<T>({x_shape, dim_1});
    mean_out = backend::reshape<T>(mean_, x_shape);
    var_out = backend::reshape<T>(var_, x_shape);
  } else {
    std::vector<int64_t> res_shape{x.shape().at(0), groups};
    mean_out = reshape<T>(mean_, res_shape);
    var_out = reshape<T>(var_, res_shape);
  }
  if (need_cast) {
    out = cast<T>(out, org_dtype);
  }

  return std::make_tuple(out, mean_out, var_out);
}

template <typename T>
Tensor square_decomp(const Tensor& x) {
  auto org_dtype = x.dtype();
  auto x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_cast = cast<T>(x, DataType::FLOAT32);
  }

  Tensor two;
  two = full<T>(empty_shape, 2, x_cast.dtype());

  auto ans = elementwise_pow<T>(x_cast, two);
  if (need_cast) {
    return cast<T>(ans, org_dtype);
  } else {
    return ans;
  }
}

template <typename T>
Tensor embedding_decomp(const Tensor& x,
                        const Tensor& weight,
                        const int64_t padding_idx,
                        const bool sparse) {
  if (weight.dims().size() != 2) {
    PADDLE_THROW(phi::errors::Unimplemented("Only support weight with 2-D."));
  }

  const int64_t NoPadding = -1;
  Tensor weight_tmp = weight;
  Tensor res;
  if (has_dynamic_shape(x.shape())) {
    if (padding_idx != NoPadding) {
      Tensor put_shape = shape<T>(sum<T>(weight, {0}, weight.dtype(), true));
      Tensor padding_idx_tensor =
          backend::full_with_tensor<T>(put_shape, padding_idx, DataType::INT64);
      Tensor zeros =
          backend::full_with_tensor<T>(put_shape, 0.0, weight.dtype());
      weight_tmp = put_along_axis<T>(weight, padding_idx_tensor, zeros, 0);
    }

    if (x.dims().size() <= 1) {
      res = gather<T>(weight_tmp, x);
      if (x.dims().size() == 0) {
        res = squeeze<T>(res, {0});
      }
    } else {
      std::vector<int64_t> tar_shape{-1, 1};
      auto x_reshape = reshape<T>(x, tar_shape);
      auto out = gather<T>(weight_tmp, x_reshape);
      auto x_t_shape = shape<T>(x);
      auto token_dim = get_slice<T>(shape<T>(out), 1);
      auto res_t_shape = concat<T>({x_t_shape, token_dim}, 0);
      res = backend::reshape<T>(out, res_t_shape);
    }
  } else {
    if (padding_idx != NoPadding) {
      std::vector<int64_t> put_shape{1, weight.dims()[1]};
      Tensor padding_idx_tensor =
          full<T>(put_shape, padding_idx, DataType::INT64);
      Tensor zeros = full<T>(put_shape, 0.0, weight.dtype());
      weight_tmp = put_along_axis<T>(weight, padding_idx_tensor, zeros, 0);
    }

    if (x.dims().size() <= 1) {
      res = gather<T>(weight_tmp, x);
      if (x.dims().size() == 0) {
        res = std::get<0>(squeeze_decomp<T>(res, {0}));
      }
    } else {
      std::vector<int64_t> tar_shape{-1, 1};
      auto x_reshape = reshape<T>(x, tar_shape);
      auto out = gather<T>(weight_tmp, x_reshape);

      auto res_dims = x.shape();
      res_dims.push_back(-1);
      res = reshape<T>(out, res_dims);
    }
  }
  if (res.dtype() != weight.dtype()) {
    res = cast<T>(res, weight.dtype());
  }
  return res;
}

template <typename T>
Tensor index_sample_decomp(const Tensor& x, const Tensor& index) {
  std::vector<int64_t> tmp_shape{-1, 1};
  auto index_dim = get_slice<T>(shape<T>(index), 0);
  auto start = full<T>({1}, 0, index_dim.dtype());
  auto step = full<T>({1}, 1, index_dim.dtype());
  auto arange_tmp = reshape<T>(
      backend::arange_with_tensor<T>(start, index_dim, step, index.dtype()),
      tmp_shape);

  auto index_res = reshape<T>(
      backend::expand_with_tensor<T>(arange_tmp, shape<T>(index)), tmp_shape);
  auto index_ = reshape<T>(index, tmp_shape);
  auto concat_res = concat<T>({index_res, index_}, 1);
  auto res = backend::reshape<T>(gather_nd<T>(x, concat_res), shape<T>(index));

  if (res.dtype() != x.dtype()) {
    return cast<T>(res, x.dtype());
  } else {
    return res;
  }
}

template <typename T>
Tensor elu_decomp(const Tensor& x, const float alpha) {
  auto org_dtype = x.dtype();
  auto x_cast = x;

  bool need_cast = is_half_dtype(org_dtype);
  if (need_cast) {
    x_cast = cast<T>(x, DataType::FLOAT32);
  }

  const Tensor zero = full<T>(x_cast.shape(), 0, x_cast.type());
  auto tmp_res = alpha * (exp<T>(x_cast) - 1);
  auto ans = where<T>(x_cast > zero, x_cast, tmp_res);
  if (need_cast) {
    return cast<T>(ans, org_dtype);
  } else {
    return ans;
  }
}

}  // namespace details

}  // namespace primitive
}  // namespace paddle
