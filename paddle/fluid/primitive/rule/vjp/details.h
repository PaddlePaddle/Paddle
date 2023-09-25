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

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include <vector>

#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"

namespace paddle {
namespace primitive {
namespace details {

template <typename T>
void divide_grad(const Tensor& x,
                 const Tensor& y,
                 const Tensor& out,
                 const Tensor& out_grad,
                 int axis,
                 Tensor* dx,
                 Tensor* dy) {
  if (dy) {
    // dy = -(x/y^2) * dout
    auto dy_res = -(x / y.pow(2.0)) * out_grad;
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dy_res, dy);
      } else {
        auto dy_reduce_res =
            sum<T>(dy_res, phi::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      }
    } else {
      set_output<T>(dy_res, dy);
    }
  }  // indicate we will compute dy
  if (dx) {
    // dx = (1/y) * dout
    auto one_tensor = full<T>(phi::vectorize(y.dims()), 1.0, y.dtype());
    auto dx_res = one_tensor / y * out_grad;
    if (y.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), y.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dx_res, dx);
      } else {
        auto dx_reduce_res =
            sum<T>(dx_res, phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      }

    } else {
      set_output<T>(dx_res, dx);
    }
  }  // indicate we will compute dx
}

template <typename T>
void sum_grad(const Tensor& x,
              const Tensor& out_grad,
              const IntArray& axis,
              bool keepdim,
              bool reduce_all,
              Tensor* x_grad) {
  if (!x_grad) {
    return;
  }
  std::vector<int64_t> x_dim = phi::vectorize<int64_t>(x.dims());
  int64_t axis_size = axis.size();
  int64_t x_dim_size = x_dim.size();
  reduce_all = false;
  if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
    reduce_all = true;
  } else {
    reduce_all = false;
  }
  auto x_grad_tmp = Tensor();
  if (x_dim_size == 1) {
    x_grad_tmp = expand<T>(out_grad, IntArray(x_dim));
  } else {
    if (!keepdim) {
      auto axis_ = std::vector<int64_t>();
      if (reduce_all) {
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
      auto out_grad_shape = get_unsqueeze_dims(out_grad, axis_);
      auto out_grad_ = reshape<T>(out_grad, out_grad_shape);
      x_grad_tmp = expand<T>(out_grad_, IntArray(x_dim));
    } else {
      x_grad_tmp = expand<T>(out_grad, IntArray(x_dim));
    }
  }

  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void gelu_grad(const Tensor& x,
               const Tensor& out_grad,
               bool approximate,
               Tensor* x_grad) {
  if (!x_grad) return;
  // Promote to fp32 when the input type is fp16 for keeping consistent with
  // phi kernel

  if (x.dtype() == phi::DataType::FLOAT16 ||
      x.dtype() == phi::DataType::BFLOAT16) {
    auto promoted_x = cast<T>(x, phi::DataType::FLOAT32);
    auto promoted_out_grad = cast<T>(out_grad, phi::DataType::FLOAT32);
    if (approximate) {
      float kbeta = M_SQRT2 * M_2_SQRTPI * 0.5;
      float kkappa = 0.044715;
      auto x_sq = promoted_x * promoted_x;
      auto x_cube = x_sq * promoted_x;
      auto inner = kbeta * (promoted_x + kkappa * x_cube);
      auto tanh_inner = tanh<T>(inner);

      auto left = scale<T>(promoted_x, 0.5);
      auto right = scale<T>(tanh_inner, 1., 1.);

      auto left_derivative = scale<T>(right, 0.5);

      auto tanh_derivative = scale<T>(tanh_inner * tanh_inner, -1., 1.);
      auto inner_derivative = kbeta * (scale<T>(3 * kkappa * x_sq, 1., 1.));
      auto right_derivative = left * tanh_derivative * inner_derivative;

      set_output<T>(
          cast<T>(promoted_out_grad * (left_derivative + right_derivative),
                  x.type()),
          x_grad);
    } else {
      float kalpha = M_SQRT1_2;
      float kbeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
      auto cdf = scale<T>(scale<T>(erf<T>(kalpha * promoted_x), 1., 1.), 0.5);
      auto pdf = kbeta * exp<T>(scale<T>(promoted_x * promoted_x, -0.5));
      set_output<T>(
          cast<T>(promoted_out_grad * (cdf + promoted_x * pdf), x.type()),
          x_grad);
    }
  } else {
    // Scale only support fp32 attr in static graph mode, use elementwise_xx
    // when precision is over fp32.
    if (approximate) {
      auto kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
      auto kKappa = 0.044715;
      auto x_sq = x * x;
      auto x_cube = x_sq * x;
      auto inner = kBeta * (x + kKappa * x_cube);
      auto tanh_inner = tanh<T>(inner);

      auto left = scale<T>(x, 0.5);
      auto right = scale<T>(tanh_inner, 1., 1.);

      auto left_derivative = scale<T>(right, 0.5);

      auto tanh_derivative = scale<T>(tanh_inner * tanh_inner, -1., 1.);
      auto inner_derivative = kBeta * (scale<T>(3 * kKappa * x_sq, 1., 1.));
      auto right_derivative = left * tanh_derivative * inner_derivative;

      set_output<T>(out_grad * (left_derivative + right_derivative), x_grad);
    } else {
      auto kAlpha = M_SQRT1_2;
      auto kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
      auto cdf = scale<T>(scale<T>(erf<T>(kAlpha * x), 1., 1.), 0.5);
      auto pdf = kBeta * exp<T>(scale<T>(x * x, -0.5));
      set_output<T>(out_grad * (cdf + x * pdf), x_grad);
    }
  }
}

template <typename T>
void reshape_grad(const Tensor& x, const Tensor& grad_out, Tensor* grad_x) {
  if (grad_x) {
    auto grad_x_tmp = reshape<T>(grad_out, phi::vectorize(x.dims()));
    set_output<T>(grad_x_tmp, grad_x);
  }
}

template <typename T>
void transpose_grad(const Tensor& grad_out,
                    const std::vector<int>& perm,
                    Tensor* grad_x) {
  if (grad_x) {
    std::vector<int> reverse_perm(perm);
    // make origin ranks
    for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
      if (perm[i] >= 0) {
        reverse_perm[perm[i]] = i;
      } else {
        reverse_perm[perm[i] + perm.size()] = i;
      }
    }
    auto grad_x_tmp = transpose<T>(grad_out, reverse_perm);
    set_output<T>(grad_x_tmp, grad_x);
  }
}

template <typename T>
void tanh_grad(const Tensor& out, const Tensor& grad_out, Tensor* grad_x) {
  if (!grad_x) return;
  auto grad_x_tmp = grad_out * (1 - out * out);
  set_output<T>(grad_x_tmp, grad_x);
}

template <typename T>
void concat_grad(const std::vector<Tensor>& x,
                 const Tensor& out_grad,
                 const Scalar& axis,
                 std::vector<Tensor*> x_grad) {
  int axis_value = axis.to<int>();
  int rank = x[0].dims().size();
  if (axis_value < 0) {
    axis_value = axis_value + rank;
  }
  axis_value = axis_value > 0 ? axis_value : 0;
  std::vector<int> sections;
  int x_num = x.size();
  for (int i = 0; i < x_num; ++i) {
    sections.push_back(x[i].dims()[axis_value]);
  }
  std::vector<Tensor> x_grad_tmp =
      split<T>(out_grad, IntArray(sections), axis_value);
  for (int i = 0; i < x_num; ++i) {
    if (x_grad[i]) {
      set_output<T>(x_grad_tmp.at(i), x_grad.at(i));
    }
  }
}

template <typename T>
void split_grad(const std::vector<Tensor>& out_grad,
                const Scalar& axis,
                Tensor* x_grad) {
  if (x_grad) {
    auto grad = concat<T>(out_grad, axis);
    set_output<T>(grad, x_grad);
  }
}

template <typename T>
void cast_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto res = cast<T>(out_grad, x.dtype());
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void add_grad(const Tensor& x,
              const Tensor& y,
              const Tensor& out_grad,
              int axis,
              Tensor* dx,
              Tensor* dy) {
  if (dy) {
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(out_grad, dy);
      } else {
        auto dy_reduce_res =
            out_grad.sum(phi::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      }

    } else {
      by_pass<T>(out_grad, dy);
    }
  }
  if (dx) {
    if (y.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), y.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(out_grad, dx);
      } else {
        auto dx_reduce_res =
            out_grad.sum(phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      }
    } else {
      by_pass<T>(out_grad, dx);
    }
  }
}

template <typename T>
void multiply_grad(const Tensor& x,
                   const Tensor& y,
                   const Tensor& out_grad,
                   int axis,
                   Tensor* x_grad,
                   Tensor* y_grad) {
  if (x_grad) {
    auto x_grad_unreduce = out_grad * y;
    if (x_grad_unreduce.dims() != x.dims()) {
      auto axes = get_reduce_dims_from_out(x_grad_unreduce.dims(), x.dims());
      if (!axes.size()) {
        set_output<T>(x_grad_unreduce, x_grad);
      } else {
        auto x_grad_reduced = x_grad_unreduce.sum(
            phi::vectorize(axes), x_grad_unreduce.dtype(), false);
        if (x_grad_reduced.dims().size() != x.dims().size()) {
          x_grad_reduced = reshape<T>(x_grad_reduced, x.shape());
        }
        set_output<T>(x_grad_reduced, x_grad);
      }
    } else {
      set_output<T>(x_grad_unreduce, x_grad);
    }
  }
  if (y_grad) {
    auto y_grad_unreduce = out_grad * x;
    if (y_grad_unreduce.dims() != y.dims()) {
      auto axes = get_reduce_dims_from_out(y_grad_unreduce.dims(), y.dims());
      if (!axes.size()) {
        set_output<T>(y_grad_unreduce, y_grad);
      } else {
        auto y_grad_reduced = y_grad_unreduce.sum(
            phi::vectorize(axes), y_grad_unreduce.dtype(), false);
        if (y_grad_reduced.dims().size() != y.dims().size()) {
          y_grad_reduced = reshape<T>(y_grad_reduced, y.shape());
        }
        set_output<T>(y_grad_reduced, y_grad);
      }
    } else {
      set_output<T>(y_grad_unreduce, y_grad);
    }
  }
}

template <typename T>
void elementwise_pow_grad(const Tensor& x,
                          const Tensor& y,
                          const Tensor& out_grad,
                          Tensor* dx,
                          Tensor* dy) {
  if (dy) {
    // dy = lnx * x^y
    auto lnx = log<T>(x);
    auto x_pow_y = elementwise_pow<T>(x, y);
    auto dy_res = lnx * x_pow_y * out_grad;
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dy_res, dy);
      } else {
        auto dy_reduce_res =
            dy_res.sum(phi::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      }
    } else {
      set_output<T>(dy_res, dy);
    }
  }  // indicate we will compute dy
  if (dx) {
    // dx = y * x^(y-1)
    auto tmp_z = y - 1.0;
    auto x_pow_z = elementwise_pow<T>(x, tmp_z);
    auto dx_res = y * x_pow_z * out_grad;
    if (y.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), y.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dx_res, dx);
      } else {
        auto dx_reduce_res =
            dx_res.sum(phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      }

    } else {
      set_output<T>(dx_res, dx);
    }
  }  // indicate we will compute dx
}

template <typename T>
void layer_norm_grad(const Tensor& x,
                     const paddle::optional<Tensor>& scale,
                     const paddle::optional<Tensor>& bias,
                     const Tensor& mean,
                     const Tensor& variance,
                     const Tensor& out_grad,
                     float epsilon,
                     int begin_norm_axis,
                     Tensor* x_grad,
                     Tensor* scale_grad,
                     Tensor* bias_grad) {
  auto x_dims = x.dims();
  auto shape_1 = 1;  // front part
  auto shape_2 = 1;  // back part
  for (int i = 0; i < begin_norm_axis; ++i) {
    shape_1 *= x_dims[i];
  }
  for (int i = begin_norm_axis; i < x.dims().size(); ++i) {
    shape_2 *= x_dims[i];
  }
  auto scale_ptr = scale.get_ptr();
  auto bias_ptr = bias.get_ptr();

  auto x_cast = reshape<T>(x, std::vector<int64_t>({shape_1, shape_2}));
  auto out_grad_cast =
      reshape<T>(out_grad, std::vector<int64_t>({shape_1, shape_2}));
  auto mean_ = reshape<T>(mean, std::vector<int64_t>({shape_1, 1}));
  auto variance_ = reshape<T>(variance, std::vector<int64_t>({shape_1, 1}));

  Tensor scale_cast;
  if (scale_ptr) {
    scale_cast = reshape<T>(*scale_ptr, std::vector<int64_t>({1, shape_2}));
  }

  // cast dtype to float32 if dtype =float16 or bfloat16

  auto x_sub_mean = x_cast - mean_;          // M,N
  auto tmp = (1.0 / (variance_ + epsilon));  // M,1
  // auto sqrt_var_1 = sqrt<T>(tmp);            // M,1
  auto sqrt_var_1 = elementwise_pow<T>(
      tmp, full<T>(phi::vectorize(tmp.dims()), 0.5, tmp.dtype()));
  auto x_sub_mean_mul_sqrt_var_1 = x_sub_mean * sqrt_var_1;

  if (x_grad) {
    auto out_grad_scale = out_grad_cast;  // M,N
    if (scale_ptr) {
      out_grad_scale = out_grad_cast * scale_cast;  // M,N * 1,N = M,N
    }

    auto dx_end = sqrt_var_1 * out_grad_scale;
    auto d_mean =
        dx_end.sum(std::vector<int64_t>({1}), x_cast.dtype(), true);  // M,1

    auto d_std_1 =
        (tmp * x_sub_mean * out_grad_scale)
            .sum(std::vector<int64_t>({1}), x_cast.dtype(), true);  // M,1
    auto d_std = d_std_1 * x_sub_mean_mul_sqrt_var_1;  // M,1 * M,N = M,N

    auto d_mean_d_std = (1.0 / shape_2) * (d_mean + d_std);
    auto x_grad_tmp = dx_end - d_mean_d_std;
    x_grad_tmp = reshape<T>(x_grad_tmp, phi::vectorize(x.dims()));

    set_output<T>(x_grad_tmp, x_grad);
  }

  if (scale_grad) {
    if (scale_ptr) {
      auto scale_grad_tmp =
          (x_sub_mean_mul_sqrt_var_1 * out_grad_cast)
              .sum(std::vector<int64_t>({0}), x_cast.dtype(), true);
      scale_grad_tmp = reshape<T>(scale_grad_tmp, scale_ptr->shape());
      set_output<T>(scale_grad_tmp, scale_grad);
    } else {
      scale_grad = nullptr;
    }
  }

  if (bias_grad) {
    if (bias_ptr) {
      auto bias_grad_tmp =
          out_grad_cast.sum(std::vector<int64_t>({0}), x_cast.dtype(), true);
      bias_grad_tmp = reshape<T>(bias_grad_tmp, bias_ptr->shape());
      set_output<T>(bias_grad_tmp, bias_grad);
    } else {
      bias_grad = nullptr;
    }
  }
}

template <typename T>
void dropout_grad(const Tensor& mask,
                  const Tensor& out_grad,
                  const Scalar& p,
                  bool is_test,
                  const std::string& mode,
                  Tensor* x_grad) {
  if (!x_grad) return;
  if (is_test) {
    if (mode == "upscale_in_train") {
      by_pass<T>(out_grad, x_grad);
    } else {
      set_output<T>(out_grad * (1.0 - p.to<float>()), x_grad);
    }
  } else {
    if (mode == "upscale_in_train") {
      if (p.to<float>() == 1.0f) {
        set_output<T>(scale<T>(out_grad, 0.0), x_grad);
      } else {
        set_output<T>(scale<T>(out_grad * cast<T>(mask, out_grad.dtype()),
                               1.0 / (1.0 - p.to<float>())),
                      x_grad);
      }
    } else {
      set_output<T>(out_grad * cast<T>(mask, out_grad.dtype()), x_grad);
    }
  }
}

}  // namespace details
}  // namespace primitive
}  // namespace paddle
