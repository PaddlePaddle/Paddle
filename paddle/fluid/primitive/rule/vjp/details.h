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
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"

namespace paddle {
namespace primitive {
namespace details {

template <typename T>
void abs_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto sign_tmp = sign<T>(x);
    set_output<T>(out_grad * sign_tmp, x_grad);
  }
}

template <typename T>
void assign_grad(const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    by_pass<T>(out_grad, x_grad);
  }
}

template <typename T>
void cumsum_grad(const Tensor& x,
                 const Tensor& out_grad,
                 const Scalar& axis,
                 bool flatten,
                 bool exclusive,
                 bool reverse,
                 Tensor* x_grad) {
  if (x_grad) {
    auto grad = cumsum<T>(out_grad, axis, flatten, exclusive, !reverse);
    grad = reshape<T>(grad, x.shape());
    set_output<T>(grad, x_grad);
  }
}

template <typename T>
void cumprod_grad(const Tensor& x,
                  const Tensor& out,
                  const Tensor& out_grad,
                  int dim,
                  bool exclusive,
                  bool reverse,
                  Tensor* x_grad) {
  if (x_grad) {
    // dx = cumsum(out * out_grad, dim, false, exclusive, !reverse) / x
    std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
    auto zero_tensor = full<T>(x_dim, 0.0, x.dtype());
    auto zero_mask = cast<T>(equal<T>(x, zero_tensor), x.dtype());
    // determine the index of first zero
    auto zero_mask_cumsum_exclusive =
        cumsum<T>(zero_mask, dim, false, true, reverse);
    auto zero_mask_cumsum = scale<T>(zero_mask_cumsum_exclusive, 2) + zero_mask;
    auto ones_tensor = full<T>(x_dim, 1.0, x.dtype());
    auto first_zero_mask =
        cast<T>(equal<T>(zero_mask_cumsum, ones_tensor), x.dtype());
    // compute the grad for position with value not equal to 0
    auto common_dx = cumsum<T>(out * out_grad, dim, false, exclusive, !reverse);
    // fill the positions of 0 with 1.
    auto replace_one = (1 - zero_mask) * x + zero_mask;
    // fill the first positions of 0 with 1.
    auto replace_first_one = (1 - first_zero_mask) * x + first_zero_mask;
    // recompute the grad of the first position with 0
    auto cumprod_recompute =
        cumprod<T>(replace_first_one, dim, exclusive, reverse);
    auto zeros_dx = cumsum<T>(
        cumprod_recompute * out_grad, dim, false, exclusive, !reverse);
    auto x_grad_res =
        ((1 - first_zero_mask) * common_dx + first_zero_mask * zeros_dx) /
        replace_one;
    set_output<T>(x_grad_res, x_grad);
  }
}

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
    auto dy_res = -(x / (y * y)) * out_grad;
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape())) {
      auto dy_tmp = reduce_as<T>(dy_res, y);
      set_output<T>(dy_tmp, dy);
    } else {
      if (out_grad.dims() != y.dims()) {
        phi::DDim reduce_dim =
            get_reduce_dims_from_out(out_grad.dims(), y.dims());
        auto dy_reduce_res =
            sum<T>(dy_res, common::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      } else {
        set_output<T>(dy_res, dy);
      }
    }
  }  // indicate we will compute dy
  if (dx) {
    // dx = (1/y) * dout
    Tensor one_tensor = full_scalar<T>(1.0, y.dtype());
    auto dx_res = one_tensor / y * out_grad;
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape())) {
      auto dx_tmp = reduce_as<T>(dx_res, x);
      set_output<T>(dx_tmp, dx);
    } else {
      if (out_grad.dims() != x.dims()) {
        auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), x.dims());
        auto dx_reduce_res =
            sum<T>(dx_res, common::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      } else {
        set_output<T>(dx_res, dx);
      }
    }
  }  // indicate we will compute dx
}

template <typename T>
void floor_grad(const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto zero_tensor =
        full<T>(common::vectorize(out_grad.dims()), 0.0, out_grad.dtype());
    set_output<T>(zero_tensor, x_grad);
  }
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

  int64_t axis_size = axis.size();
  int64_t x_dim_size = x.dims().size();
  auto x_grad_tmp = Tensor();
  reduce_all = false;
  if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
    reduce_all = true;
  } else {
    reduce_all = false;
  }
  if (has_dynamic_shape(x.shape())) {
    Tensor x_shape = shape<T>(x);
    if (x_dim_size == 1) {
      x_grad_tmp = backend::expand<T>(out_grad, x_shape);
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
        Tensor out_grad_shape = shape<T>(out_grad);
        size_t total_shape_size = out_grad.shape().size() + axis_.size();
        std::vector<Tensor> result_shape;
        size_t j = 0, k = 0;
        Tensor ones = full<T>({1}, 1, x_shape.dtype());
        for (size_t i = 0; i < total_shape_size; i++) {
          if (j < axis_.size() && axis_[j] == int64_t(i)) {
            result_shape.push_back(ones);
            j++;
          } else {
            result_shape.push_back(get_slice<T>(out_grad_shape, int64_t(k)));
            k++;
          }
        }
        auto out_grad_ = backend::reshape<T>(out_grad, concat<T>(result_shape));
        x_grad_tmp = backend::expand<T>(out_grad_, x_shape);
      } else {
        x_grad_tmp = backend::expand<T>(out_grad, x_shape);
      }
    }
  } else {
    std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
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
  }
  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void mean_grad(const Tensor& x,
               const Tensor& out_grad,
               const IntArray& axis,
               bool keepdim,
               bool reduce_all,
               Tensor* x_grad) {
  if (!x_grad) {
    return;
  }
  Tensor x_grad_tmp;
  sum_grad<T>(x, out_grad, axis, keepdim, reduce_all, &x_grad_tmp);

  Tensor div_factor = [&] {
    Tensor factor_tensor;
    auto axis_data = axis.GetData();
    const std::vector<int64_t> x_dim = x.shape();
    if (axis.size() == 0) {
      for (size_t i = 0; i < x_dim.size(); ++i) {
        axis_data.push_back(i);
      }
    }
    if (has_dynamic_shape(x_dim, axis_data)) {
      auto x_shape = shape<T>(x);
      factor_tensor =
          slice<T>(x_shape, {0}, {axis_data[0]}, {axis_data[0] + 1}, {1}, {0});
      for (size_t i = 1; i < axis_data.size(); ++i) {
        factor_tensor =
            factor_tensor *
            slice<T>(
                x_shape, {0}, {axis_data[i]}, {axis_data[i] + 1}, {1}, {0});
      }
      factor_tensor = cast<T>(factor_tensor, x.dtype());
    } else {
      int64_t factor = 1;
      for (int64_t idx : axis_data) {
        if (idx < 0) idx += x_dim.size();
        factor *= x_dim[idx];
      }
      factor_tensor = full<T>(std::vector<int64_t>{}, factor, x.dtype());
    }
    return factor_tensor;
  }();

  set_output<T>(x_grad_tmp / div_factor, x_grad);
}

template <typename T>
void gelu_grad(const Tensor& x,
               const Tensor& out_grad,
               bool approximate,
               Tensor* x_grad) {
  if (!x_grad) return;
  // Promote to fp32 when the input type is fp16 for keeping consistent with
  // phi kernel

  if (is_half_dtype(x.dtype())) {
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
void reduce_as_grad(const Tensor& x,
                    const Tensor& target,
                    const Tensor& out_grad,
                    Tensor* x_grad) {
  if (!x_grad) {
    return;
  }
  std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
  std::vector<int64_t> axis = common::vectorize<int64_t>(
      get_reduce_dims_from_out(x.dims(), target.dims()));
  int64_t axis_size = axis.size();
  if (axis_size == 0) {
    by_pass<T>(out_grad, x_grad);
    return;
  }
  int64_t x_dim_size = x_dim.size();

  auto x_grad_tmp = Tensor();
  if (x_dim_size == 1) {
    x_grad_tmp = expand<T>(out_grad, IntArray(x_dim));
  } else {
    auto axis_ = std::vector<int64_t>();
    for (int64_t i = 0; i < axis_size; i++) {
      axis_.push_back(axis[i]);
      if (axis[i] < 0) {
        axis_[i] += x_dim_size;
      }
    }
    Tensor out_grad_ = out_grad;
    if (out_grad.shape().size() != x.shape().size()) {
      auto out_grad_shape = get_unsqueeze_dims(out_grad, axis_);
      out_grad_ = reshape<T>(out_grad, out_grad_shape);
    }
    x_grad_tmp = expand<T>(out_grad_, IntArray(x_dim));
  }

  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void reshape_grad(const Tensor& xshape,
                  const Tensor& grad_out,
                  Tensor* grad_x) {
  if (grad_x) {
    // xshape: [0] + x.shape
    auto xshape_dims = xshape.dims();
    auto x_dims = common::slice_ddim(xshape_dims, 1, xshape_dims.size());
    auto grad_x_tmp = reshape<T>(grad_out, common::vectorize(x_dims));
    set_output<T>(grad_x_tmp, grad_x);
  }
}

template <typename T>
void roll_grad(const Tensor& x,
               const Tensor& out_grad,
               const IntArray& shifts,
               const std::vector<int64_t>& axis,
               Tensor* x_grad) {
  if (x_grad) {
    auto shifts_ = shifts.GetData();
    int64_t nums = shifts_.size();
    for (int64_t i = 0; i < nums; i++) {
      shifts_[i] = 0 - shifts_[i];
    }
    auto x_grad_output = roll<T>(out_grad, shifts_, axis);
    set_output<T>(x_grad_output, x_grad);
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
void scatter_grad(const Tensor& index,
                  const Tensor& updates,
                  const Tensor& out_grad,
                  bool overwrite,
                  Tensor* x_grad,
                  Tensor* updates_grad) {
  if (x_grad) {
    auto zero_tensor =
        full<T>(common::vectorize(updates.dims()), 0.0, updates.dtype());
    auto tmp_grad = scatter<T>(out_grad, index, zero_tensor, false);
    set_output<T>(tmp_grad, x_grad);
  }

  if (updates_grad) {
    Scalar tmp_zero = 0;
    auto tmp_updates_grad = gather<T>(out_grad, index, tmp_zero);
    set_output<T>(tmp_updates_grad, updates_grad);
  }
}

template <typename T>
void scatter_nd_add_grad(const Tensor& index,
                         const Tensor& updates,
                         const Tensor& out_grad,
                         Tensor* x_grad,
                         Tensor* updates_grad) {
  if (x_grad) {
    by_pass<T>(out_grad, x_grad);
  }
  if (updates_grad) {
    // Gradient by Gather: dUpdates = dO[Ids]
    auto tmp_updates_grad = gather_nd<T>(out_grad, index);
    set_output<T>(tmp_updates_grad, updates_grad);
  }
}

template <typename T>
void sin_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  auto x_grad_tmp = cos<T>(x) * out_grad;
  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void cos_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  auto x_grad_tmp = -sin<T>(x) * out_grad;
  set_output<T>(x_grad_tmp, x_grad);
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

  int x_num = x.size();
  std::vector<Tensor> x_grad_tmp;
  bool has_dynamic = false;
  for (size_t i = 0; i < x.size(); i++) {
    if (has_dynamic_shape(x[i].shape())) {
      has_dynamic = true;
      break;
    }
  }
  if (has_dynamic) {
    std::vector<Tensor> sections;
    for (int i = 0; i < x_num; i++) {
      sections.push_back(get_slice<T>(shape<T>(x[i]), int64_t(axis_value)));
    }
    Tensor sections_tensor = concat<T>(sections);
    x_grad_tmp =
        backend::split<T>(out_grad,
                          sections_tensor,
                          full<T>({1}, axis_value, sections_tensor.dtype()));
  } else {
    std::vector<int> sections;
    for (int i = 0; i < x_num; ++i) {
      sections.push_back(x[i].dims()[axis_value]);
    }
    x_grad_tmp = split<T>(out_grad, IntArray(sections), axis_value);
  }
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
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape())) {
      auto dy_tmp = reduce_as<T>(out_grad, y);
      set_output<T>(dy_tmp, dy);
    } else {
      if (out_grad.dims() != y.dims()) {
        phi::DDim reduce_dim =
            get_reduce_dims_from_out(out_grad.dims(), y.dims());
        auto dy_reduce_res =
            out_grad.sum(common::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);

      } else {
        by_pass<T>(out_grad, dy);
      }
    }
  }
  if (dx) {
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape())) {
      auto dx_tmp = reduce_as<T>(out_grad, x);
      set_output<T>(dx_tmp, dx);
    } else {
      if (out_grad.dims() != x.dims()) {
        auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), x.dims());
        auto dx_reduce_res =
            out_grad.sum(common::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      } else {
        by_pass<T>(out_grad, dx);
      }
    }
  }
}

template <typename T>
void subtract_grad(const Tensor& x,
                   const Tensor& y,
                   const Tensor& out_grad,
                   int axis,
                   Tensor* dx,
                   Tensor* dy) {
  if (dy) {
    auto scale_out_grad = scale<T>(out_grad, -1.0, 0.0, true);
    if (has_dynamic_shape(y.shape()) || has_dynamic_shape(out_grad.shape())) {
      auto dy_tmp = reduce_as<T>(scale_out_grad, y);
      set_output<T>(dy_tmp, dy);
    } else {
      if (out_grad.dims() != y.dims()) {
        phi::DDim reduce_dim =
            get_reduce_dims_from_out(out_grad.dims(), y.dims());
        auto dy_reduce_res =
            scale_out_grad.sum(common::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      } else {
        by_pass<T>(scale_out_grad, dy);
      }
    }
  }
  if (dx) {
    if (has_dynamic_shape(x.shape()) || has_dynamic_shape(out_grad.shape())) {
      auto dx_tmp = reduce_as<T>(out_grad, x);
      set_output<T>(dx_tmp, dx);
    } else {
      if (out_grad.dims() != x.dims()) {
        auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), x.dims());
        auto dx_reduce_res =
            out_grad.sum(common::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      } else {
        by_pass<T>(out_grad, dx);
      }
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
    if (has_dynamic_shape(x.shape()) ||
        has_dynamic_shape(x_grad_unreduce.shape())) {
      auto x_grad_reduced = reduce_as<T>(x_grad_unreduce, x);
      set_output<T>(x_grad_reduced, x_grad);
    } else {
      if (x_grad_unreduce.dims() != x.dims()) {
        auto axes = get_reduce_dims_from_out(x_grad_unreduce.dims(), x.dims());
        auto x_grad_reduced = x_grad_unreduce.sum(
            common::vectorize(axes), x_grad_unreduce.dtype(), false);
        if (x_grad_reduced.dims().size() != x.dims().size()) {
          x_grad_reduced = reshape<T>(x_grad_reduced, x.shape());
        }
        set_output<T>(x_grad_reduced, x_grad);
      } else {
        set_output<T>(x_grad_unreduce, x_grad);
      }
    }
  }
  if (y_grad) {
    auto y_grad_unreduce = out_grad * x;
    if (has_dynamic_shape(y.shape()) ||
        has_dynamic_shape(y_grad_unreduce.shape())) {
      auto y_grad_reduced = reduce_as<T>(y_grad_unreduce, y);
      set_output<T>(y_grad_reduced, y_grad);
    } else {
      if (y_grad_unreduce.dims() != y.dims()) {
        auto axes = get_reduce_dims_from_out(y_grad_unreduce.dims(), y.dims());
        auto y_grad_reduced = y_grad_unreduce.sum(
            common::vectorize(axes), y_grad_unreduce.dtype(), false);
        if (y_grad_reduced.dims().size() != y.dims().size()) {
          y_grad_reduced = reshape<T>(y_grad_reduced, y.shape());
        }
        set_output<T>(y_grad_reduced, y_grad);
      } else {
        set_output<T>(y_grad_unreduce, y_grad);
      }
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
    if (has_dynamic_shape(out_grad.shape()) || has_dynamic_shape(y.shape())) {
      auto dy_reduce_res = reduce_as<T>(dy_res, y);
      set_output<T>(dy_reduce_res, dy);
    } else {
      if (out_grad.dims() != y.dims()) {
        phi::DDim reduce_dim =
            get_reduce_dims_from_out(out_grad.dims(), y.dims());
        auto dy_reduce_res =
            dy_res.sum(common::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      } else {
        set_output<T>(dy_res, dy);
      }
    }
  }  // indicate we will compute dy
  if (dx) {
    // dx = y * x^(y-1)
    if (has_dynamic_shape(out_grad.shape()) || has_dynamic_shape(x.shape())) {
      Tensor one_tensor = full_scalar<T>(1.0, y.dtype());
      Tensor x_pow_z = elementwise_pow<T>(x, y - one_tensor);
      Tensor dx_res = y * x_pow_z * out_grad;
      auto dx_reduce_res = reduce_as<T>(dx_res, x);
      set_output<T>(dx_reduce_res, dx);
    } else {
      auto tmp_z = y - 1.0;
      auto x_pow_z = elementwise_pow<T>(x, tmp_z);
      auto dx_res = y * x_pow_z * out_grad;
      if (out_grad.dims() != x.dims()) {
        auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), x.dims());
        auto dx_reduce_res =
            dx_res.sum(common::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      } else {
        set_output<T>(dx_res, dx);
      }
    }
  }  // indicate we will compute dx
}

template <typename T>
void pow_grad(const Tensor& x,
              const Tensor& out_grad,
              const Scalar& y,
              Tensor* x_grad) {
  if (x_grad) {
    if (has_dynamic_shape(x.shape())) {
      Tensor y_tensor = backend::full_with_tensor<T>(shape<T>(x), y, x.dtype());
      Tensor one_tensor = full_scalar<T>(1.0, x.dtype());
      auto dx_res = y_tensor * elementwise_pow<T>(x, y - one_tensor) * out_grad;
      set_output<T>(dx_res, x_grad);
    } else {
      auto y_value = y.to<float>();
      auto dx_res = y_value * x.pow(y_value - 1) * out_grad;
      set_output<T>(dx_res, x_grad);
    }
  }
}

template <typename T>
void scale_grad(const Tensor& out_grad, const Scalar& scale, Tensor* x_grad) {
  if (x_grad) {
    auto dx_res = primitive::scale<T>(
        out_grad, scale, /*bias=*/0.0f, /*bias_after_scale=*/true);
    set_output<T>(dx_res, x_grad);
  }
}

template <typename T>
void stack_grad(const std::vector<Tensor>& x,
                const Tensor& out_grad,
                int axis,
                std::vector<Tensor*> x_grad) {
  // use rank of **stacked** tensor as len of axes
  int out_rank = out_grad.dims().size();  // len(x[0].shape)

  // ensure axis >= 0
  if (axis < 0) {
    axis = ((axis % out_rank) + out_rank) % out_rank;
  }

  // split out_grad to grads for each input tensor
  int x_num = x.size();
  std::vector<int> sections(x_num, 1);
  std::vector<Tensor> x_grad_tmp =
      split<T>(out_grad, phi::IntArray(sections), axis);

  // compose shape for each input tensor
  std::vector<int64_t> grad_shape;
  auto out_dim = out_grad.dims().size();
  for (int i = 0; i < out_dim; ++i) {
    if (i != axis) {
      grad_shape.push_back(out_grad.dims()[i]);
    }
  }

  // assign to each input tensor if need grad(stop_gradient=False)
  for (int i = 0; i < x_num; ++i) {
    if (x_grad[i]) {
      set_output<T>(reshape<T>(x_grad_tmp[i], grad_shape), x_grad[i]);
    }
  }
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
  if (x.dtype() == phi::DataType::FLOAT16 ||
      x.dtype() == phi::DataType::BFLOAT16) {
    x_cast = cast<T>(x_cast, phi::DataType::FLOAT32);
    out_grad_cast = cast<T>(out_grad_cast, phi::DataType::FLOAT32);
    if (scale_ptr) {
      scale_cast = cast<T>(scale_cast, phi::DataType::FLOAT32);
    }
  }

  auto x_sub_mean = x_cast - mean_;          // M,N
  auto tmp = (1.0 / (variance_ + epsilon));  // M,1
  auto sqrt_var_1 = sqrt<T>(tmp);            // M,1
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
    x_grad_tmp = reshape<T>(x_grad_tmp, common::vectorize(x.dims()));

    if (x.dtype() == phi::DataType::FLOAT16 ||
        x.dtype() == phi::DataType::BFLOAT16) {
      x_grad_tmp = cast<T>(x_grad_tmp, x.dtype());
    }
    set_output<T>(x_grad_tmp, x_grad);
  }

  if (scale_grad) {
    if (scale_ptr) {
      auto scale_grad_tmp =
          (x_sub_mean_mul_sqrt_var_1 * out_grad_cast)
              .sum(std::vector<int64_t>({0}), x_cast.dtype(), true);
      scale_grad_tmp = reshape<T>(scale_grad_tmp, scale_ptr->shape());
      if (scale_ptr->dtype() == phi::DataType::FLOAT16 ||
          scale_ptr->dtype() == phi::DataType::BFLOAT16) {
        scale_grad_tmp = cast<T>(scale_grad_tmp, scale_ptr->dtype());
      }
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
      if (bias_ptr->dtype() == phi::DataType::FLOAT16 ||
          bias_ptr->dtype() == phi::DataType::BFLOAT16) {
        bias_grad_tmp = cast<T>(bias_grad_tmp, bias_ptr->dtype());
      }
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

template <typename T>
void erf_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto m_2_sqrt_pi =
        full<T>(common::vectorize(x.dims()), M_2_SQRTPI, x.dtype());
    auto neg_one = full<T>(common::vectorize(x.dims()), -1.0, x.dtype());
    auto neg_tmp = neg_one * x * x;
    auto mul_tmp = m_2_sqrt_pi * exp<T>(neg_tmp);
    set_output<T>(out_grad * mul_tmp, x_grad);
  }
}

template <typename T>
void expand_grad(const Tensor& x,
                 const Tensor& out_grad,
                 const IntArray& shape,
                 Tensor* x_grad) {
  if (x_grad) {
    if (out_grad.dims() != x.dims()) {
      auto axes = get_reduce_dims_from_out(out_grad.dims(), x.dims());
      auto reduced = out_grad.sum(common::vectorize(axes), x.dtype(), false);
      if (reduced.dims().size() != x.dims().size()) {
        reduced = reshape<T>(reduced, x.shape());
      }
      set_output<T>(reduced, x_grad);
    } else {
      by_pass<T>(out_grad, x_grad);
    }
  }
}

template <typename T>
void log_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // dx = dout / x
    set_output<T>(out_grad / x, x_grad);
  }
}

template <typename T>
void square_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    Tensor x_grad_tmp = 2 * x * out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void exp_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    if (out.dtype() == phi::DataType::FLOAT16 ||
        out.dtype() == phi::DataType::BFLOAT16) {
      Tensor out_promote = cast<T>(out, phi::DataType::FLOAT32);
      Tensor out_grad_promote = cast<T>(out_grad, phi::DataType::FLOAT32);
      set_output<T>(cast<T>(out_promote * out_grad_promote, out.dtype()),
                    x_grad);
    } else {
      set_output<T>(out_grad * out, x_grad);
    }
  }
}

template <typename T>
void sqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // This calculation is important for resnet.
    auto factor = full_scalar<T>(0.5, out.dtype());
    auto x_grad_tmp = (factor / out) * out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void rsqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // This calculation is important for resnet.
    auto factor = full_scalar<T>(-0.5, out.dtype());
    auto x_grad_tmp = factor * out * out * out * out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void silu_grad(const Tensor& x,
               const Tensor& out,
               const Tensor& out_grad,
               Tensor* x_grad) {
  if (x_grad) {
    auto org_dtype = x.dtype();
    bool need_cast = org_dtype == phi::DataType::FLOAT16 ||
                     org_dtype == phi::DataType::BFLOAT16;
    if (need_cast) {
      auto x_cast = cast<T>(x, phi::DataType::FLOAT32);
      auto out_cast = cast<T>(out, phi::DataType::FLOAT32);
      auto out_grad_cast = cast<T>(out_grad, phi::DataType::FLOAT32);
      auto res = out_grad_cast * sigmoid<T>(x_cast) * (1.0 + x_cast - out_cast);
      set_output<T>(cast<T>(res, org_dtype), x_grad);
    } else {
      auto one = full_scalar<T>(1.0, x.dtype());
      auto res = out_grad * sigmoid<T>(x) * (one + x - out);
      set_output<T>(res, x_grad);
    }
  }
}

template <typename T>
void softmax_grad(const Tensor& out,
                  const Tensor& out_grad,
                  int axis,
                  Tensor* x_grad) {
  if (x_grad) {
    if (out_grad.dims().size() > 0) {
      if (axis >= 0) {
        auto new_out_grad = out_grad * out;
        auto tmp_x_grad = new_out_grad -
                          out * sum<T>(new_out_grad, {axis}, out.dtype(), true);
        set_output<T>(tmp_x_grad, x_grad);
      } else {
        auto new_out_grad = out_grad * out;
        auto tmp_x_grad =
            new_out_grad - out * sum<T>(new_out_grad,
                                        {out.dims().size() + axis},
                                        out.dtype(),
                                        true);
        set_output<T>(tmp_x_grad, x_grad);
      }
    } else {
      Tensor zeros = full_scalar<T>(0.0, out.dtype());
      set_output<T>(out_grad * zeros, x_grad);
    }
  }
}

template <typename T>
void squeeze_grad(const Tensor& xshape,
                  const Tensor& out_grad,
                  const IntArray& axis,
                  Tensor* x_grad) {
  if (x_grad) {
    auto x_grad_out = unsqueeze<T>(out_grad, axis);
    set_output<T>(x_grad_out, x_grad);
  }
}

template <typename T>
void unsqueeze_grad(const Tensor& xshape,
                    const Tensor& out_grad,
                    const IntArray& axis,
                    Tensor* x_grad) {
  // for xshape = [10, 2, 5], axis = [3, 1, 1], out_grad.shape = [10, 1, 1, 2,
  // 5, 1], it outputs squeeze axis = [5, 2, 1]
  const auto& IncreaseAxis = [](std::vector<int64_t>* axis_data,
                                int64_t pivot) {
    for (size_t i = 0; i < axis_data->size(); ++i) {
      if ((*axis_data)[i] >= pivot) (*axis_data)[i] += 1;
    }
  };
  const auto& GetRealAxis = [&](const IntArray& axis) -> decltype(auto) {
    // for axis = [0, 3, 3], it outputs [0, 3, 3+1], because unsqueeze support
    // duplicated axis.
    std::vector<int64_t> output_axis;
    const int64_t x_rank = xshape.dims().size() - 1;
    const std::vector<int64_t> axis_data = axis.GetData();
    for (size_t i = 0; i < axis_data.size(); ++i) {
      int64_t value = axis_data[i];
      if (value < 0) value += (x_rank + i + 1);
      IncreaseAxis(&output_axis, value);
      output_axis.push_back(value);
    }
    return output_axis;
  };

  if (x_grad) {
    auto x_grad_out = squeeze<T>(out_grad, GetRealAxis(axis));
    set_output<T>(x_grad_out, x_grad);
  }
}

template <typename T>
void matmul_grad(const Tensor& x,
                 const Tensor& y,
                 const Tensor& out_grad,
                 bool transpose_x,
                 bool transpose_y,
                 Tensor* x_grad,
                 Tensor* y_grad) {
  auto unsqueeze_out_grad = out_grad;
  size_t out_grad_rank = out_grad.shape().size();
  size_t x_rank = x.shape().size();
  size_t y_rank = y.shape().size();
  int temp_rank_y = out_grad_rank - 1;
  int temp_rank_x = out_grad_rank;
  if (out_grad_rank < y_rank) {
    unsqueeze_out_grad = unsqueeze<T>(out_grad, {temp_rank_y});
  }
  if (out_grad_rank < x_rank) {
    unsqueeze_out_grad = unsqueeze<T>(out_grad, {temp_rank_x});
  }

  auto temp_x_unsqueeze = x;
  if (x_rank == 1) {
    temp_x_unsqueeze = unsqueeze<T>(x, {0});
  }

  auto temp_y_unsqueeze = y;
  if (y_rank == 1) {
    temp_y_unsqueeze = unsqueeze<T>(y, {1});
  }

  if (x_grad) {
    auto x_grad_mm =
        matmul<T>(unsqueeze_out_grad, temp_y_unsqueeze, false, !transpose_y);
    auto x_grad_trans = x_grad_mm;

    if (transpose_x) {
      std::vector<int> reverse_perm;
      for (size_t i = 0; i < x_grad_trans.shape().size(); i++) {
        reverse_perm.push_back(i);
      }
      std::swap(reverse_perm[reverse_perm.size() - 1],
                reverse_perm[reverse_perm.size() - 2]);
      x_grad_trans = transpose<T>(x_grad_mm, reverse_perm);
    }

    if (x_grad_trans.dims() != x.dims()) {
      phi::DDim x_reduce_dim = get_reduce_dims_from_out(
          x_grad_trans.dims(), temp_x_unsqueeze.dims());
      auto dx_reduce_res = sum<T>(
          x_grad_trans, common::vectorize(x_reduce_dim), x.dtype(), false);
      auto x_grad_out = reshape<T>(dx_reduce_res, x.shape());
      set_output<T>(x_grad_out, x_grad);
    } else {
      auto x_grad_out = x_grad_trans;
      set_output<T>(x_grad_out, x_grad);
    }
  }

  if (y_grad) {
    auto y_grad_mm =
        matmul<T>(temp_x_unsqueeze, unsqueeze_out_grad, !transpose_x, false);
    auto y_grad_trans = y_grad_mm;

    if (transpose_y) {
      std::vector<int> reverse_perm;
      for (size_t i = 0; i < y_grad_mm.shape().size(); i++) {
        reverse_perm.push_back(i);
      }
      std::swap(reverse_perm[reverse_perm.size() - 1],
                reverse_perm[reverse_perm.size() - 2]);
      y_grad_trans = transpose<T>(y_grad_mm, reverse_perm);
    }

    if (y_grad_trans.dims() != y.dims()) {
      phi::DDim y_reduce_dim = get_reduce_dims_from_out(
          y_grad_trans.dims(), temp_y_unsqueeze.dims());
      auto dy_reduce_res = sum<T>(
          y_grad_trans, common::vectorize(y_reduce_dim), y.dtype(), false);
      auto y_grad_out = reshape<T>(dy_reduce_res, y.shape());
      set_output<T>(y_grad_out, y_grad);
    } else {
      auto y_grad_out = y_grad_trans;
      set_output<T>(y_grad_out, y_grad);
    }
  }
}

template <typename T>
void maximum_grad(const Tensor& x,
                  const Tensor& y,
                  const Tensor& out_grad,
                  Tensor* x_grad,
                  Tensor* y_grad) {
  if (x_grad) {
    auto x_tmp = cast<T>(greater_than<T>(x, y), out_grad.dtype());
    auto dx_res = out_grad * x_tmp;
    if (out_grad.dims() != x.dims()) {
      auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), x.dims());
      auto dx_reduce_res =
          dx_res.sum(common::vectorize(reduce_dim), x.dtype(), false);
      auto dx_tmp = reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
      set_output<T>(dx_tmp, x_grad);
    } else {
      set_output<T>(dx_res, x_grad);
    }
  }

  if (y_grad) {
    auto y_tmp = cast<T>(less_equal<T>(x, y), out_grad.dtype());
    auto dy_res = out_grad * y_tmp;
    if (out_grad.dims() != y.dims()) {
      phi::DDim reduce_dim =
          get_reduce_dims_from_out(out_grad.dims(), y.dims());
      auto dy_reduce_res =
          dy_res.sum(common::vectorize(reduce_dim), y.dtype(), false);
      auto dy_tmp = reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
      set_output<T>(dy_tmp, y_grad);
    } else {
      set_output<T>(dy_res, y_grad);
    }
  }
}

template <typename T>
void masked_select_grad(const Tensor& x,
                        const Tensor& mask,
                        const Tensor& out_grad,
                        Tensor* x_grad) {
  if (x_grad) {
    auto promoted_x = x;
    auto promoted_out_grad = out_grad;
    if (is_half_dtype(x.dtype())) {
      promoted_x = cast<T>(x, DataType::FLOAT32);
      promoted_out_grad = cast<T>(out_grad, DataType::FLOAT32);
    }

    auto x_num = 1;
    for (size_t i = 0; i < promoted_x.shape().size(); i++) {
      x_num *= promoted_x.shape()[i];
    }

    auto grad_num = 1;
    for (size_t i = 0; i < promoted_out_grad.shape().size(); i++) {
      grad_num *= promoted_out_grad.shape()[i];
    }

    auto end = full<T>({1}, x_num, x.dtype());
    auto start = full<T>({1}, 0, x.dtype());
    auto step = full<T>({1}, 1, x.dtype());
    auto x_arange =
        backend::arange_with_tensor<T>(start, end, step, promoted_x.dtype());

    auto x_arange_reshape = reshape<T>(x_arange, promoted_x.shape());

    auto x_index = masked_select<T>(x_arange_reshape, mask);

    auto index_num = x_index.shape()[0];

    auto grad_reshape =
        cast<T>(reshape<T>(promoted_out_grad, {grad_num}), promoted_x.dtype());

    auto grad_trans = grad_reshape;
    if (grad_num > index_num) {
      grad_trans = slice<T>(grad_reshape, {0}, {0}, {index_num}, {1}, {});
    } else if (grad_num < index_num) {
      auto pad_zeros = full<T>({index_num - grad_num}, 0, promoted_x.dtype());
      grad_trans = concat<T>({grad_reshape, pad_zeros}, 0);
    }

    auto input_tensor = full<T>({x_num}, 0, promoted_x.dtype());
    auto index_tensor = cast<T>(x_index, DataType::INT64);
    auto update_tensor = grad_trans;
    auto x_output =
        scatter<T>(input_tensor, index_tensor, update_tensor, false);
    auto res = cast<T>(reshape<T>(x_output, promoted_x.shape()), x.dtype());
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void relu_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    Tensor zeros = full_scalar<T>(0.0, out.dtype());
    auto mask = greater_than<T>(out, zeros);
    auto res = cast<T>(mask, out.dtype()) * out_grad;
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void gather_grad(const Tensor& x,
                 const Tensor& index,
                 const Tensor& out_grad,
                 const Scalar& axis,
                 Tensor* grad_x) {
  auto zero_tensor = full<T>(common::vectorize(x.dims()), 0.0, x.dtype());
  std::vector<int> tmp_perm;

  // change axis to rank 0
  int axis_value = axis.to<int>();
  tmp_perm.push_back(axis_value);
  // make other ranks
  for (int i = 0; i < x.dims().size(); ++i) {
    if (i != axis_value) {
      tmp_perm.push_back(i);
    }
  }
  std::vector<int> reverse_perm(tmp_perm);
  // make origin ranks
  for (int i = 0; i < static_cast<int>(tmp_perm.size()); ++i) {
    if (tmp_perm[i] >= 0) {
      reverse_perm[tmp_perm[i]] = i;
    } else {
      reverse_perm[tmp_perm[i] + tmp_perm.size()] = i;
    }
  }

  // transpose out_grad and zero grad to target rank.
  auto tmp_zero_x_grad = zero_tensor;
  auto tmp_out_grad = out_grad;
  if (zero_tensor.dims().size() > 0) {
    tmp_zero_x_grad = transpose<T>(zero_tensor, tmp_perm);
  }
  if (out_grad.dims().size() > 0) {
    tmp_out_grad = transpose<T>(out_grad, tmp_perm);
  }
  // scatter grad to grad_x
  auto tmp_grad_x = scatter<T>(tmp_zero_x_grad, index, tmp_out_grad, false);
  auto tmp_grad_x_transposed = tmp_grad_x;
  if (tmp_grad_x.dims().size() > 0) {
    tmp_grad_x_transposed = transpose<T>(tmp_grad_x, reverse_perm);
  }
  set_output<T>(tmp_grad_x_transposed, grad_x);
}

template <typename T>
void gather_nd_grad(const Tensor& x,
                    const Tensor& index,
                    const Tensor& out_grad,
                    Tensor* x_grad) {
  if (x_grad) {
    auto zero_tensor = full<T>(common::vectorize(x.dims()), 0.0, x.dtype());
    auto x_grad_tmp = scatter_nd_add<T>(zero_tensor, index, out_grad);
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void instance_norm_grad(const Tensor& x,
                        const paddle::optional<Tensor>& scale,
                        const Tensor& saved_mean,
                        const Tensor& saved_variance,
                        const Tensor& y_grad,
                        float epsilon,
                        Tensor* x_grad,
                        Tensor* scale_grad,
                        Tensor* bias_grad) {
  const int n = x.dims()[0];
  const int c = x.dims()[1];
  const int h = x.dims()[2];
  const int w = x.dims()[3];

  auto promoted_y_grad = y_grad;
  if (x.dtype() == phi::DataType::FLOAT16 ||
      x.dtype() == phi::DataType::BFLOAT16) {
    promoted_y_grad = cast<T>(y_grad, phi::DataType::FLOAT32);
  }

  Tensor x_hat;
  Tensor std_inv;
  if (scale_grad || x_grad) {
    auto promoted_x = x;
    auto promoted_saved_mean = saved_mean;
    auto promoted_saved_var = saved_variance;
    if (x.dtype() == phi::DataType::FLOAT16 ||
        x.dtype() == phi::DataType::BFLOAT16) {
      promoted_x = cast<T>(x, phi::DataType::FLOAT32);
      promoted_saved_mean = cast<T>(saved_mean, phi::DataType::FLOAT32);
      promoted_saved_var = cast<T>(saved_variance, phi::DataType::FLOAT32);
    }
    auto mean = reshape<T>(promoted_saved_mean, IntArray({n, c, 1, 1}))
                    .tile(IntArray({1, 1, h, w}));
    std_inv = reshape<T>(promoted_saved_var, IntArray({n, c, 1, 1}))
                  .tile(IntArray({1, 1, h, w}));
    x_hat = (promoted_x - mean) * std_inv;
  }

  // x_grad = scale * inv_var * (y_grad - y_grad.mean(2,3) - x_hat * (y_grad *
  // x_hat).mean((h,w)))
  if (x_grad) {
    auto scale_data =
        reshape<T>(scale.get_ptr() ? scale.get()
                                   : full<T>(IntArray({c}), 1., x.dtype()),
                   IntArray({1, c, 1, 1}))
            .tile(IntArray({n, 1, h, w}));
    auto promoted_scale = scale_data;
    if (scale_data.dtype() == phi::DataType::FLOAT16 ||
        scale_data.dtype() == phi::DataType::BFLOAT16) {
      promoted_scale = cast<T>(scale_data, phi::DataType::FLOAT32);
    }
    auto result =
        (promoted_scale * std_inv) *
        (promoted_y_grad -
         promoted_y_grad.sum(IntArray({2, 3}), promoted_y_grad.dtype(), true) /
             (h * w) -
         (x_hat * ((promoted_y_grad * x_hat)
                       .sum(IntArray({2, 3}), promoted_y_grad.dtype(), true) /
                   (h * w))));
    if (x.dtype() == phi::DataType::FLOAT16 ||
        x.dtype() == phi::DataType::BFLOAT16) {
      set_output<T>(cast<T>(result, x.dtype()), x_grad);
    } else {
      set_output<T>(result, x_grad);
    }
  }
  // scale_grad = x_hat * y_grad.sum(n, h, w)
  if (scale_grad) {
    auto result = (promoted_y_grad * x_hat).sum(IntArray({0, 2, 3}));
    auto scale_dtype = scale.get_ptr() ? scale.get().dtype() : x.dtype();
    if (scale_dtype == phi::DataType::FLOAT16 ||
        scale_dtype == phi::DataType::BFLOAT16) {
      set_output<T>(cast<T>(result, scale_dtype), scale_grad);
    } else {
      set_output<T>(result, scale_grad);
    }
  }
  // d_bias = y_grad.sum(n, h, w)
  if (bias_grad) {
    auto result = promoted_y_grad.sum(IntArray({0, 2, 3}));
    auto scale_dtype = scale.get_ptr() ? scale.get().dtype() : x.dtype();
    if (scale_dtype == phi::DataType::FLOAT16 ||
        scale_dtype == phi::DataType::BFLOAT16) {
      set_output<T>(cast<T>(result, scale_dtype), bias_grad);
    } else {
      set_output<T>(result, bias_grad);
    }
  }
}

template <typename T>
void pad_grad(const Tensor& input,
              const Tensor& out_grad,
              const std::vector<int>& paddings,
              const Scalar& pad_value,
              Tensor* input_grad) {
  if (input_grad) {
    size_t rank = input.dims().size();
    auto out_dims = out_grad.dims();

    std::vector<int64_t> starts(rank, 0);
    std::vector<int64_t> ends(rank, 0);
    std::vector<int64_t> axes(rank, 0);
    std::vector<int64_t> infer_flags(rank, 1);
    std::vector<int64_t> decrease_axis({});
    for (size_t i = 0; i < rank; ++i) {
      starts[i] = static_cast<int64_t>(paddings[2 * i]);
      ends[i] = static_cast<int64_t>(out_dims[i] - paddings[2 * i + 1]);
      axes[i] = i;
    }
    auto out_tmp =
        slice<T>(out_grad, axes, starts, ends, infer_flags, decrease_axis);
    set_output<T>(out_tmp, input_grad);
  }
}

template <typename T>
void max_grad(const Tensor& x,
              const Tensor& out,
              const Tensor& out_grad,
              const IntArray& axis,
              bool keepdim,
              bool reduce_all,
              Tensor* x_grad) {
  if (!x_grad) {
    return;
  }
  auto zero_tensor = full<T>(common::vectorize(x.dims()), 0.0, x.dtype());
  std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
  int64_t axis_size = axis.size();
  int64_t x_dim_size = x_dim.size();
  reduce_all = false;
  if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
    reduce_all = true;
  } else {
    reduce_all = false;
  }
  auto x_grad_tmp = Tensor();
  if (x_dim_size == 0 || x_dim_size == 1 || keepdim) {
    auto out_grad_tmp = out_grad.expand(IntArray(x_dim));
    auto out_tmp = out.expand(IntArray(x_dim));
    auto mask = equal<T>(x, out_tmp);
    x_grad_tmp = where<T>(mask, out_grad_tmp, zero_tensor);
  } else {
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
    auto out_ = reshape<T>(out, out_grad_shape);
    auto out_grad_tmp = out_grad_.expand(IntArray(x_dim));
    auto out_tmp = out_.expand(IntArray(x_dim));
    auto mask = equal<T>(x, out_tmp);
    x_grad_tmp = where<T>(mask, out_grad_tmp, zero_tensor);
  }
  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void slice_grad(const Tensor& input,
                const Tensor& out_grad,
                const std::vector<int64_t>& axes,
                const IntArray& starts,
                const IntArray& ends,
                const std::vector<int64_t>& infer_flags,
                const std::vector<int64_t>& decrease_axis,
                Tensor* input_grad) {
  if (input_grad) {
    size_t rank = input.dims().size();
    auto out_dims = out_grad.dims();
    std::vector<int64_t> origin_out_shape;
    auto in_dims = input.dims();

    auto decrease_size = decrease_axis.size();
    if (decrease_size > 0) {
      if (decrease_size == static_cast<size_t>(in_dims.size())) {
        // all dims decrease
        out_dims = common::make_ddim(std::vector<int>(decrease_size, 1));
      } else {
        origin_out_shape.resize(out_dims.size() + decrease_size, -1);
        for (size_t i = 0; i < decrease_size; ++i) {
          origin_out_shape[decrease_axis[i]] = 1;
        }

        int index = 0;
        for (size_t i = 0; i < origin_out_shape.size(); ++i) {
          if (origin_out_shape[i] == -1) {
            origin_out_shape[i] = out_dims[index];
            ++index;
          }
        }
        out_dims = common::make_ddim(origin_out_shape);
      }
    }

    std::vector<int> offsets(rank, 0);
    std::vector<int> extents(rank, 0);
    for (size_t i = 0; i < rank; ++i) {
      offsets[i] = 0;
      extents[i] = out_dims[i];
    }
    for (size_t i = 0; i < axes.size(); ++i) {
      int axis = axes[i];
      int64_t start = starts[i] < 0 ? (starts[i] + in_dims[axis]) : starts[i];
      start = std::max(start, static_cast<int64_t>(0));
      offsets[axis] = start;
    }

    std::vector<int> paddings;
    for (size_t i = 0; i < rank; ++i) {
      paddings.push_back(offsets[i]);
      paddings.push_back((in_dims[i] - out_dims[i]) - offsets[i]);
    }
    Tensor reshape_out_grad;
    if (out_grad.shape().size() == 0) {
      reshape_out_grad = full<T>({1}, 1, input.dtype());
    } else {
      reshape_out_grad = out_grad;
    }

    if (decrease_size > 0 &&
        (decrease_size != static_cast<size_t>(in_dims.size()))) {
      auto out_tmp =
          pad<T>(reshape<T>(reshape_out_grad, origin_out_shape), paddings, 0.0);
      set_output<T>(out_tmp, input_grad);
    } else {
      auto out_tmp = pad<T>(reshape_out_grad, paddings, 0.0);
      set_output<T>(out_tmp, input_grad);
    }
  }
}

template <typename T>
void tile_grad(const Tensor& x,
               const Tensor& out_grad,
               const IntArray& repeat_times,
               Tensor* x_grad) {
  if (x_grad) {
    auto repeat_times_data = repeat_times.GetData();
    auto out_grad_shape = common::vectorize<int>(out_grad.dims());
    auto result = out_grad;
    for (int i = 0; i < static_cast<int>(repeat_times_data.size()); i++) {
      int size = out_grad_shape[i] / repeat_times_data[i];
      std::vector<int> sections(repeat_times_data[i], size);
      auto split_arr = split<T>(result, IntArray(sections), i);
      result = full<T>(common::vectorize(split_arr[0].dims()), 0.0, x.dtype());
      for (int j = 0; j < static_cast<int>(split_arr.size()); j++) {
        result = split_arr[j] + result;
      }
    }
    result = reshape<T>(result, x.shape());
    set_output<T>(result, x_grad);
  }
}

template <typename T>
void hardswish_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto offset = full<T>(common::vectorize(x.dims()), 3.0, x.dtype());
    auto condition = less_equal<T>(x, offset);
    auto tmp1 = where<T>(condition, out_grad * ((x / 3.0) + 0.5), out_grad);
    auto res = where<T>(
        less_than<T>(x, full<T>(common::vectorize(x.dims()), -3.0, x.dtype())),
        full<T>(common::vectorize(x.dims()), 0.0, x.dtype()),
        tmp1);
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void leaky_relu_grad(const Tensor& out,
                     const Tensor& out_grad,
                     float negative_slope,
                     Tensor* x_grad) {
  if (x_grad) {
    auto condition = greater_than<T>(
        out, full<T>(common::vectorize(out.dims()), 0.0, out.dtype()));
    auto res = where<T>(condition, out_grad, out_grad * negative_slope);
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void sigmoid_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto one_tensor = full_scalar<T>(1.0, out.dtype());
    set_output<T>(out_grad * (out * (one_tensor - out)), x_grad);
  }
}

template <typename T>
void topk_grad(const Tensor& x,
               const Tensor& indices,
               const Tensor& out_grad,
               const Scalar& k,
               const int& axis,
               const bool& largest,
               const bool& sorted,
               Tensor* x_grad) {
  if (x_grad) {
    // put_along_axis doesn't support zero dim
    if (x.dims().size() == 0) {
      by_pass<T>(out_grad, x_grad);
      return;
    }
    auto zero_tensor = full<T>(common::vectorize(x.dims()), 0, x.dtype());
    auto x_grad_tmp = put_along_axis<T>(zero_tensor, indices, out_grad, axis);
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void batch_norm_grad(const Tensor& x,
                     const paddle::optional<Tensor>& scale,
                     const paddle::optional<Tensor>& bias,
                     const paddle::optional<Tensor>& mean_out,
                     const paddle::optional<Tensor>& variance_out,
                     const Tensor& saved_mean,
                     const Tensor& saved_variance,
                     const paddle::optional<Tensor>& reserve_space,
                     const Tensor& out_grad,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout,
                     bool is_test,
                     bool use_global_stats,
                     bool trainable_statistics,
                     Tensor* x_grad,
                     Tensor* scale_grad,
                     Tensor* bias_grad) {
  use_global_stats = is_test || use_global_stats;

  DataLayout data_layout_ = common::StringToDataLayout(data_layout);

  Tensor x_data = x;
  Tensor out_grad_data = out_grad;

  bool need_cast = x.dtype() == phi::DataType::FLOAT16 ||
                   x.dtype() == phi::DataType::BFLOAT16;
  if (need_cast) {
    x_data = cast<T>(x, phi::DataType::FLOAT32);
  }
  if (out_grad.dtype() == phi::DataType::FLOAT16 ||
      out_grad.dtype() == phi::DataType::BFLOAT16) {
    out_grad_data = cast<T>(out_grad, phi::DataType::FLOAT32);
  }

  auto x_dims = x_data.dims();
  const int C = (data_layout_ == DataLayout::kNCHW ? x_dims[1]
                                                   : x_dims[x_dims.size() - 1]);
  int nume = 1;
  for (auto i = 0; i < x_dims.size(); i++) {
    nume = nume * x_dims[i];
  }

  const int nhw = nume / C;

  if (x_dims.size() == 2 && data_layout_ == DataLayout::kNCHW) {
    data_layout_ = DataLayout::kNHWC;
  }

  auto run_var = variance_out.get();
  auto run_mean = mean_out.get();

  Tensor mean_data;
  Tensor rsqrt_var;

  if (use_global_stats) {
    auto eps =
        full<T>(common::vectorize(run_var.dims()), epsilon, run_var.dtype());
    mean_data = run_mean;
    rsqrt_var = rsqrt<T>(run_var + eps);
  } else {
    mean_data = saved_mean;
    rsqrt_var = saved_variance;
  }

  // inv_var = 1 / sqrt(var + eps)
  // reduce_axis = [0, 2, 3] (NCHW) [0, 1, 2] (NHWC)
  //
  // d_bias = np.sum(d_y, reduce_axis)
  // d_scale = np.sum((X - mean) / inv_var * dy, reduce_axis)
  //
  // train mode
  // d_x = (1. / nhw) * scale * inv_var
  // *(nhw * d_y - np.sum(d_y, reduce_axis) - (X - mean) * inv_var * inv_var *
  // np.sum(d_y * (X - mean), reduce_axis))
  //
  // test mode
  // d_x = d_y * scale * inv_var

  std::vector<int> nchw_to_nhwc_dim = {0, 2, 3, 1};
  std::vector<int> nhwc_to_nchw_dim = {0, 3, 1, 2};
  auto reduce_axis = IntArray(std::vector<int64_t>{0, 1, 2});
  auto dtype = x_data.dtype();

  switch (data_layout_) {
    case DataLayout::kNCHW: {
      auto nhwc_x = transpose<T>(x_data, nchw_to_nhwc_dim);
      auto nhwc_out_grad = transpose<T>(out_grad_data, nchw_to_nhwc_dim);
      auto nhwc_out_grad_sum = sum<T>(nhwc_out_grad, reduce_axis, dtype, false);

      auto sum_dout_mul_diff = sum<T>(
          nhwc_out_grad * (nhwc_x - mean_data), reduce_axis, dtype, false);

      if (x_grad) {
        if (use_global_stats) {
          auto nhwc_x_grad = rsqrt_var * nhwc_out_grad;
          if (scale) {
            nhwc_x_grad = scale.get() * nhwc_x_grad;
          }
          auto nchw_x_grad = transpose<T>(nhwc_x_grad, nhwc_to_nchw_dim);
          if (need_cast) {
            nchw_x_grad = cast<T>(nchw_x_grad, x.dtype());
          }
          set_output<T>(nchw_x_grad, x_grad);
        } else {
          auto part1 = rsqrt_var;
          if (scale) {
            part1 = scale.get() * part1;
          }
          auto mean_temp1 = nhwc_out_grad_sum / nhw;
          auto mean_temp2 = sum_dout_mul_diff / nhw * rsqrt_var * rsqrt_var;
          auto part2 =
              nhwc_out_grad - mean_temp1 - (nhwc_x - mean_data) * mean_temp2;

          auto x_grad_data = part1 * part2;
          auto nchw_x_grad = transpose<T>(x_grad_data, nhwc_to_nchw_dim);
          if (need_cast) {
            nchw_x_grad = cast<T>(nchw_x_grad, x.dtype());
          }
          set_output<T>(nchw_x_grad, x_grad);
        }
      }
      if (scale_grad) {
        auto scale_grad_data = sum_dout_mul_diff * rsqrt_var;
        set_output<T>(scale_grad_data, scale_grad);
      }
      if (bias_grad) {
        set_output<T>(assign<T>(nhwc_out_grad_sum), bias_grad);
      }
      break;
    }
    case DataLayout::kNHWC: {
      if (x_grad) {
        auto out_grad_data_sum =
            sum<T>(out_grad_data, reduce_axis, dtype, false);
        auto nhwc_sum_dout_mul_diff = sum<T>(
            out_grad_data * (x_data - mean_data), reduce_axis, dtype, false);
        if (use_global_stats) {
          auto x_grad_data = rsqrt_var * out_grad_data;
          if (scale) {
            x_grad_data = scale.get() * x_grad_data;
          }
          if (need_cast) {
            x_grad_data = cast<T>(x_grad_data, x.dtype());
          }
          set_output<T>(x_grad_data, x_grad);
        } else {
          auto part1 = rsqrt_var;
          if (scale) {
            part1 = scale.get() * part1;
          }
          auto mean_temp1 = out_grad_data_sum / nhw;
          auto mean_temp2 =
              nhwc_sum_dout_mul_diff / nhw * rsqrt_var * rsqrt_var;
          auto part2 =
              out_grad_data - mean_temp1 - (x_data - mean_data) * mean_temp2;

          auto x_grad_data = part1 * part2;
          if (need_cast) {
            x_grad_data = cast<T>(x_grad_data, x.dtype());
          }
          set_output<T>(x_grad_data, x_grad);
        }
        if (scale_grad) {
          auto scale_grad_data = nhwc_sum_dout_mul_diff * rsqrt_var;
          set_output<T>(scale_grad_data, scale_grad);
        }
        if (bias_grad) {
          set_output<T>(assign<T>(out_grad_data_sum), bias_grad);
        }
      }
      break;
    }

    default:
      PADDLE_THROW(phi::errors::InvalidArgument("Unknown storage order: %s",
                                                data_layout));
  }
}

template <typename T>
void prod_grad(const Tensor& x,
               const Tensor& out,
               const Tensor& out_grad,
               const IntArray& axis,
               bool keep_dim,
               bool reduce_all,
               Tensor* x_grad) {
  if (x_grad) {
    std::vector<int64_t> x_dim = common::vectorize<int64_t>(x.dims());
    int64_t axis_size = axis.size();
    int64_t x_dim_size = x_dim.size();
    reduce_all = false;
    if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
      reduce_all = true;
    } else {
      reduce_all = false;
    }
    auto out_grad_tmp = Tensor();
    auto x_reshape = Tensor();
    std::vector<int64_t> unchange_axis, change_axis, transpose_shape,
        cumprod_shape;
    std::vector<int> transpose_dim, origin_position;
    if (x_dim_size == 1) {
      out_grad_tmp = out_grad.expand(IntArray(x_dim));
    } else {
      if (!keep_dim) {
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
        out_grad_tmp = out_grad_.expand(IntArray(x_dim));
      } else {
        out_grad_tmp = out_grad.expand(IntArray(x_dim));
      }
    }
    auto axis_ = std::vector<int64_t>();
    if (reduce_all) {
      int64_t numel = 1;
      for (int64_t i = 0; i < x_dim_size; i++) {
        axis_.push_back(i);
        numel *= x_dim[i];
      }
      cumprod_shape.push_back(numel);
      x_reshape = reshape<T>(x, cumprod_shape);
      auto left_cumprod = cumprod<T>(x_reshape, -1, true, false);
      auto right_cumprod = cumprod<T>(x_reshape, -1, true, true);
      auto x_grad_tmp = left_cumprod * right_cumprod;
      auto x_grad_tmp2 = reshape<T>(x_grad_tmp, x.shape());
      auto x_grad_res = x_grad_tmp2 * out_grad_tmp;
      set_output<T>(x_grad_res, x_grad);
    } else {
      int64_t unchange_size = x_dim_size - axis_size;
      int64_t unchange_index = 0;
      for (int64_t i = 0; i < axis_size; i++) {
        if (axis[i] < 0) {
          axis_.push_back(axis[i] + x_dim_size);
        } else {
          axis_.push_back(axis[i]);
        }
      }
      for (int64_t i = 0; i < x_dim_size; i++) {
        auto it = find(axis_.begin(), axis_.end(), i);
        if (it != axis_.end()) {
          int64_t index = it - axis_.begin();
          origin_position.push_back(static_cast<int>(unchange_size + index));
        } else {
          unchange_axis.push_back(i);
          origin_position.push_back(static_cast<int>(unchange_index));
          unchange_index += 1;
        }
      }
      int64_t numel = 1;
      for (int64_t i = 0; i < unchange_size; i++) {
        transpose_shape.push_back(x_dim[unchange_axis[i]]);
        cumprod_shape.push_back(x_dim[unchange_axis[i]]);
        transpose_dim.push_back(static_cast<int>(unchange_axis[i]));
      }
      for (int64_t i = 0; i < axis_size; i++) {
        transpose_shape.push_back(x_dim[axis_[i]]);
        transpose_dim.push_back(static_cast<int>(axis_[i]));
        numel *= x_dim[axis_[i]];
      }
      cumprod_shape.push_back(numel);
      auto x_transpose = transpose<T>(x, transpose_dim);
      x_reshape = reshape<T>(x_transpose, cumprod_shape);
      auto left_cumprod = cumprod<T>(x_reshape, -1, true, false);
      auto right_cumprod = cumprod<T>(x_reshape, -1, true, true);
      auto x_grad_tmp = left_cumprod * right_cumprod;
      auto x_grad_reshape = reshape<T>(x_grad_tmp, transpose_shape);
      auto x_grad_tmp2 = transpose<T>(x_grad_reshape, origin_position);
      auto x_grad_res = x_grad_tmp2 * out_grad_tmp;
      set_output<T>(x_grad_res, x_grad);
    }
  }
}

template <typename T>
void minimum_grad(const Tensor& x,
                  const Tensor& y,
                  const Tensor& out_grad,
                  Tensor* x_grad,
                  Tensor* y_grad) {
  if (x_grad) {
    auto x_tmp = cast<T>(less_than<T>(x, y), out_grad.dtype());
    auto dx_res = out_grad * x_tmp;
    if (out_grad.dims() != x.dims()) {
      auto reduce_dim = get_reduce_dims_from_out(out_grad.dims(), x.dims());
      auto dx_reduce_res =
          dx_res.sum(common::vectorize(reduce_dim), x.dtype(), false);
      auto dx_tmp = reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
      set_output<T>(dx_tmp, x_grad);
    } else {
      set_output<T>(dx_res, x_grad);
    }
  }

  if (y_grad) {
    auto y_tmp = cast<T>(greater_equal<T>(x, y), out_grad.dtype());
    auto dy_res = out_grad * y_tmp;
    if (out_grad.dims() != y.dims()) {
      phi::DDim reduce_dim =
          get_reduce_dims_from_out(out_grad.dims(), y.dims());
      auto dy_reduce_res =
          dy_res.sum(common::vectorize(reduce_dim), y.dtype(), false);
      auto dy_tmp = reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
      set_output<T>(dy_tmp, y_grad);
    } else {
      set_output<T>(dy_res, y_grad);
    }
  }
}

template <typename T>
void group_norm_grad(const Tensor& x,
                     const paddle::optional<Tensor>& scale,
                     const paddle::optional<Tensor>& bias,
                     const Tensor& y,
                     const Tensor& mean,
                     const Tensor& variance,
                     const Tensor& out_grad,
                     float epsilon,
                     int groups,
                     const std::string& data_layout,
                     Tensor* x_grad,
                     Tensor* scale_grad,
                     Tensor* bias_grad) {
  // x.shape=[n,c,h,w]
  // y.shape=[n,c,h,w]
  // g_size = c/g
  // scale.shape=[c]
  // mean, var: shape=[n, g]
  // inv_std = rsqrt(var + epsilon)
  // ds = sum(dy * x, axes=(2,3))
  // db = sum(dy, axes=(2,3))
  //
  // cal d_x:
  // s = g / (h*w*c)
  // if scale:
  //  ds_val = sum((ds * scale).reshape(n, g, g_size), axes=2)
  //  db_val = sum((db * scale).reshape(n, g, g_size), axes=2)
  //  p1 = (inv_std.reshape(n, g, 1)) * (scale.reshape(1, g, g_size))
  // else:
  //  ds_val = sum(ds.reshape(n, g, g_size), axes=2)
  //  db_val = sum(db.reshape(n, g, g_size), axes=2)
  //  p1 = (inv_std.reshape(n, g, 1)) * (ones(1, g, g_size))
  // p2 = (db_val * mean - ds_val) * inv_std * inv_std * inv_std * s
  // p3 = -p2 * mean - db_val * inv_std * s
  // p1.reshape(n, g, g_size, 1)
  // p2.reshape(n, g, 1, 1)
  // p3.reshape(n, g, 1, 1)
  // d_x = dy.reshape(n, g, g_size, h*w) * p1 + x.reshape(n, g, g_size, h*w)* p2
  // + p3
  //
  // cal d_scale:
  // temp = ds.reshape(n, g, g_size) - db.reshape(n, g, g_size) *
  // mean.reshape(n, g, 1)
  // d_scale = sum(temp * inv_std.reshape(n, g, 1), axes=0).reshape(c)
  //
  // cal d_bias:
  // d_bias = sum(dy, axes=(0,2,3))
  DataLayout data_layout_ = common::StringToDataLayout(data_layout);
  std::vector<int64_t> x_dims = x.shape();
  int rank = x_dims.size();
  if (rank < 3 || rank > 5) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support NCHW and NHWC format in rank {3, 4, 5}."));
  }
  int N = x_dims[0];
  int C;
  int hw = 1;
  std::vector<int64_t> reduce_axis;

  if (data_layout_ == DataLayout::kNCHW) {
    C = x_dims[1];
    for (int i = 2; i < rank; ++i) {
      hw *= x_dims[i];
      reduce_axis.push_back(i);
    }
  } else if (data_layout_ == DataLayout::kNHWC) {
    C = x_dims[rank - 1];
    for (int i = 1; i < (rank - 1); ++i) {
      hw *= x_dims[i];
      reduce_axis.push_back(i);
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("Unsupported storage order: %s",
                                              data_layout));
  }

  int g_num = C / groups;

  Tensor x_data = x;
  Tensor out_grad_data = out_grad;

  if (x.dtype() == phi::DataType::FLOAT16 ||
      x.dtype() == phi::DataType::BFLOAT16) {
    x_data = cast<T>(x, phi::DataType::FLOAT32);
  }

  if (out_grad.dtype() == phi::DataType::FLOAT16 ||
      out_grad.dtype() == phi::DataType::BFLOAT16) {
    out_grad_data = cast<T>(out_grad, phi::DataType::FLOAT32);
  }

  auto shape_group = std::vector<int64_t>({N, groups, g_num});

  std::vector<int64_t> whole_group_shape;
  if (data_layout_ == DataLayout::kNCHW) {
    whole_group_shape = std::vector<int64_t>({N, groups, g_num, -1});
  } else {
    whole_group_shape = std::vector<int64_t>({N, -1, groups, g_num});
  }
  auto var_eps = variance + epsilon;

  auto inv_std = rsqrt<T>(var_eps);

  auto inv_std_mul_s = inv_std / hw / g_num;
  auto dtype = x_data.dtype();
  auto sum_y_grad_mul_x =
      sum<T>(out_grad_data * x_data, reduce_axis, dtype, false);
  auto sum_y_grad = sum<T>(out_grad_data, reduce_axis, dtype, false);

  Tensor scale_data;
  if (scale) {
    scale_data = scale.get();
  }
  Tensor bias_data;
  if (bias) {
    bias_data = bias.get();
  }

  if (x_grad) {
    Tensor d1;
    Tensor d2;
    Tensor p1;
    if (scale) {
      if (scale_data.dtype() == phi::DataType::FLOAT16 ||
          scale_data.dtype() == phi::DataType::BFLOAT16) {
        scale_data = cast<T>(scale_data, phi::DataType::FLOAT32);
      }
      d1 = (reshape<T>(sum_y_grad_mul_x * scale_data, shape_group))
               .sum(std::vector<int64_t>({2}), dtype, false);
      d2 = (reshape<T>(sum_y_grad * scale_data, shape_group))
               .sum(std::vector<int64_t>({2}), dtype, false);
      p1 = reshape<T>(inv_std, std::vector<int64_t>({N, groups, 1})) *
           reshape<T>(scale_data, std::vector<int64_t>({1, groups, g_num}));
    } else {
      d1 = (reshape<T>(sum_y_grad_mul_x, shape_group)).sum({2}, dtype, false);
      d2 = (reshape<T>(sum_y_grad, shape_group)).sum({2}, dtype, false);
      p1 = (reshape<T>(inv_std, {N, groups, 1}))
               .expand(shape_group);  // [n, g, g_n]
    }

    auto p2 = (d2 * mean - d1) * (inv_std_mul_s / var_eps);  // [n, g]
    auto p3 = -p2 * mean - d2 * inv_std_mul_s;
    std::vector<int64_t> first_shape;
    std::vector<int64_t> second_shape;
    if (data_layout_ == DataLayout::kNCHW) {
      first_shape = get_unsqueeze_dims(p1, {3});      // [n, g, g_n, 1]
      second_shape = get_unsqueeze_dims(p2, {2, 3});  // [n, g, 1, 1]
    } else {
      first_shape = get_unsqueeze_dims(p1, {1});      // [n, 1, g, g_n]
      second_shape = get_unsqueeze_dims(p2, {1, 3});  // [n, 1, g, 1]
    }

    p1 = reshape<T>(p1, first_shape);
    p2 = reshape<T>(p2, second_shape);
    p3 = reshape<T>(p3, second_shape);
    auto tmp_1 =
        reshape<T>(out_grad_data, whole_group_shape) * p1;  // [n, hw, g, g_n]
    auto tmp_2 = reshape<T>(x_data, whole_group_shape) * p2 + p3;
    auto x_grad_data = tmp_1 + tmp_2;
    x_grad_data = reshape<T>(x_grad_data, x.shape());
    if (x.dtype() == phi::DataType::FLOAT16 ||
        x.dtype() == phi::DataType::BFLOAT16) {
      x_grad_data = cast<T>(x_grad_data, x.dtype());
    }

    set_output<T>(x_grad_data, x_grad);
  }

  if (scale_grad) {
    if (scale) {
      auto third_shape = get_unsqueeze_dims(mean, {2});
      auto tmp1 = (reshape<T>(sum_y_grad_mul_x, shape_group) -
                   reshape<T>(sum_y_grad, shape_group) *
                       reshape<T>(mean, third_shape)) *
                  reshape<T>(inv_std, third_shape);
      auto scale_grad_tmp =
          reshape<T>(tmp1.sum({0}, scale->dtype(), false), {C});
      set_output<T>(scale_grad_tmp, scale_grad);
    }
  }

  if (bias_grad) {
    if (bias) {
      auto bias_grad_tmp = sum_y_grad.sum({0}, bias->dtype(), false);
      set_output<T>(bias_grad_tmp, bias_grad);
    }
  }
}

template <typename T>
void swiglu_grad(const Tensor& x,
                 const paddle::optional<Tensor>& y,
                 const Tensor& dz,
                 Tensor* dx,
                 Tensor* dy) {
  const auto& x_shape = x.shape();
  auto one_tensor = full<T>(x_shape, 1.0, x.dtype());
  Tensor x_grad;
  if (y) {
    const auto& y_tensor = y.get();
    Tensor sig = sigmoid<T>(x);
    Tensor tmp = sig * x;
    x_grad = dz * y_tensor * sig * (one_tensor + x - tmp);
    Tensor y_grad = dz * tmp;
    set_output<T>(y_grad, dy);
  } else {
    int axis = x.shape().size() - 1;
    int num = 2;
    std::vector<Tensor> xs = backend::split_with_num<T>(x, num, axis);
    Tensor sig = sigmoid<T>(xs[0]);
    Tensor tmp = sig * xs[0];
    Tensor x0_grad = dz * xs[1] * sig * (one_tensor + xs[0] - tmp);
    Tensor x1_grad = dz * tmp;
    int64_t c_axis = x_shape.size() - 1;
    x_grad = concat<T>({x0_grad, x1_grad}, c_axis);
  }
  set_output<T>(x_grad, dx);
}

}  // namespace details
}  // namespace primitive
}  // namespace paddle
