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

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>

#include "paddle/common/ddim.h"
#include "paddle/fluid/prim/api/all.h"
#include "paddle/fluid/prim/api/composite_backward/composite_double_backward_api.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/int_array.h"

namespace paddle {
namespace prim {
using Tensor = paddle::Tensor;
using IntArray = paddle::experimental::IntArrayBase<paddle::Tensor>;
//  This function should have as same signature as phi, which defined in
//  paddle/phi/api/backward/backward_api.h

template <typename T>
void pow_grad(const Tensor& x,
              const Tensor& out_grad,
              const Scalar& y,
              Tensor* x_grad) {
  // dx = y * x^(y-1) * out_grad
  if (x_grad) {
    auto y_value = y.to<float>();
    auto dx_res = y_value * x.pow(y_value - 1) * out_grad;
    set_output<T>(dx_res, x_grad);
  }  // indicate we will compute dx
}

template <typename T>
void hardswish_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto offset =
        full<T>(common::vectorize(x.dims()), 3.0, x.dtype(), x.place());
    auto condition = less_equal<T>(x, offset);
    auto tmp1 = where<T>(condition, out_grad * ((x / 3.0) + 0.5), out_grad);
    auto res = where<T>(
        less_than<T>(
            x,
            full<T>(common::vectorize(x.dims()), -3.0, x.dtype(), x.place())),
        full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place()),
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
        out,
        full<T>(common::vectorize(out.dims()), 0.0, out.dtype(), out.place()));
    auto res = where<T>(condition, out_grad, out_grad * negative_slope);
    set_output<T>(res, x_grad);
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
      auto sigmoid = 1.0 / (1.0 + exp<T>(-x_cast));
      auto res = out_grad_cast * sigmoid * (1.0 + x_cast - out_cast);
      set_output<T>(cast<T>(res, org_dtype), x_grad);
    } else {
      auto sigmoid = 1.0 / (1.0 + exp<T>(-x));
      auto res = out_grad * sigmoid * (1.0 + x - out);
      set_output<T>(res, x_grad);
    }
  }
}

template <typename T>
void relu_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto mask = greater_than<T>(
        out,
        full<T>(common::vectorize(out.dims()), 0.0, out.dtype(), out.place()));
    auto res = cast<T>(mask, out.dtype()) * out_grad;
    set_output<T>(res, x_grad);
  }
}

template <typename T>
void softmax_grad(const Tensor& out,
                  const Tensor& out_grad,
                  int axis,
                  Tensor* x_grad) {
  if (x_grad) {
    if (axis < 0) {
      axis += out.dims().size();
    }
    if (out_grad.dims().size() > 0) {
      // dx = dy * y - y * (dy*y).sum(axis)
      auto new_out_grad = out_grad * out;
      auto tmp_x_grad =
          new_out_grad - out * sum<T>(new_out_grad, {axis}, out.dtype(), true);
      set_output<T>(tmp_x_grad, x_grad);
    } else {
      set_output<T>(full<T>(common::vectorize(out_grad.dims()),
                            0.0,
                            out_grad.dtype(),
                            out_grad.place()),
                    x_grad);
    }
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
void gather_grad(const Tensor& x,
                 const Tensor& index,
                 const Tensor& out_grad,
                 const Scalar& axis,
                 Tensor* grad_x) {
  auto zero_tensor =
      full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place());
  std::vector<int> tmp_perm;

  // change axis to rank 0
  int axis_value = axis.to<int>();
  int rank = x.dims().size();
  if (axis_value < 0) {
    axis_value += rank;
  }
  tmp_perm.push_back(axis_value);
  // make other ranks
  for (int i = 0; i < rank; ++i) {
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
void tanh_grad(const Tensor& out, const Tensor& grad_out, Tensor* grad_x) {
  if (!grad_x) return;
  auto grad_x_tmp = grad_out * (1 - out * out);
  set_output<T>(grad_x_tmp, grad_x);
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
  std::vector<int64_t> axis =
      common::vectorize<int64_t>(get_reduce_dims(x.dims(), target.dims()));
  int64_t axis_size = axis.size();
  int64_t x_dim_size = x_dim.size();
  bool reduce_all = false;
  if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
    reduce_all = true;
  } else {
    reduce_all = false;
  }
  auto x_grad_tmp = Tensor();
  if (x_dim_size == 1) {
    x_grad_tmp = expand<T>(out_grad, IntArray(x_dim));
  } else {
    auto axis_ = std::vector<int64_t>();
    if (reduce_all) {
      for (int64_t i = 0; i < x_dim_size; i++) {
        axis_.push_back(i);
      }
    } else {
      for (int64_t i = 0; i < axis_size; i++) {
        axis_.push_back(axis[i]);
        if (axis[i] < 0) {
          axis_[i] += x_dim_size;
        }
      }
    }
    auto out_grad_shape = get_unsqueeze_dims(out_grad, axis_);
    auto out_grad_ = reshape<T>(out_grad, out_grad_shape);
    x_grad_tmp = expand<T>(out_grad_, IntArray(x_dim));
  }

  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void reshape_grad(const Tensor& x, const Tensor& grad_out, Tensor* grad_x) {
  if (grad_x) {
    auto grad_x_tmp = reshape<T>(grad_out, common::vectorize(x.dims()));
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
void subtract_grad(const Tensor& x,
                   const Tensor& y,
                   const Tensor& out_grad,
                   int axis,
                   Tensor* dx,
                   Tensor* dy) {
  if (dy) {
    auto scale_out_grad = scale<T>(out_grad, -1.0, 0.0, true);
    if (out_grad.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(scale_out_grad, dy);
      } else {
        auto dy_reduce_res =
            scale_out_grad.sum(common::vectorize(reduce_dim),
                               y.dtype(),
                               scale_out_grad.dims().size() == y.dims().size());
        if (dy_reduce_res.dims() != y.dims()) {
          dy_reduce_res =
              reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        }
        set_output<T>(dy_reduce_res, dy);
      }
    } else {
      by_pass<T>(scale_out_grad, dy);
    }
  }
  if (dx) {
    if (out_grad.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(out_grad, dx);
      } else {
        auto dx_reduce_res =
            out_grad.sum(common::vectorize(reduce_dim),
                         x.dtype(),
                         out_grad.dims().size() == x.dims().size());
        if (dx_reduce_res.dims() != x.dims()) {
          dx_reduce_res =
              reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        }
        set_output<T>(dx_reduce_res, dx);
      }
    } else {
      by_pass<T>(out_grad, dx);
    }
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
    if (out_grad.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(out_grad, dy);
      } else {
        auto dy_reduce_res =
            out_grad.sum(common::vectorize(reduce_dim),
                         y.dtype(),
                         out_grad.dims().size() == y.dims().size());
        if (dy_reduce_res.dims() != y.dims()) {
          dy_reduce_res =
              reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        }
        set_output<T>(dy_reduce_res, dy);
      }
    } else {
      by_pass<T>(out_grad, dy);
    }
  }
  if (dx) {
    if (out_grad.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(out_grad, dx);
      } else {
        auto dx_reduce_res =
            out_grad.sum(common::vectorize(reduce_dim),
                         x.dtype(),
                         out_grad.dims().size() == x.dims().size());
        if (dx_reduce_res.dims() != x.dims()) {
          dx_reduce_res =
              reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        }
        set_output<T>(dx_reduce_res, dx);
      }
    } else {
      by_pass<T>(out_grad, dx);
    }
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
  if (x_dim_size == 1) {
    x_grad_tmp = out_grad.expand(IntArray(x_dim));
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
      x_grad_tmp = out_grad_.expand(IntArray(x_dim));
    } else {
      x_grad_tmp = out_grad.expand(IntArray(x_dim));
    }
  }

  set_output<T>(x_grad_tmp, x_grad);
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
    // dy = -(x/y^2) * dout = -out * dout / y
    auto dy_res = -out * out_grad / y;
    if (out.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), out.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dy_res, dy);
      } else {
        auto dy_reduce_res =
            dy_res.sum(common::vectorize(reduce_dim),
                       y.dtype(),
                       dy_res.dims().size() == y.dims().size());
        if (dy_reduce_res.dims() != y.dims()) {
          dy_reduce_res =
              reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        }
        set_output<T>(dy_reduce_res, dy);
      }
    } else {
      set_output<T>(dy_res, dy);
    }
  }  // indicate we will compute dy
  if (dx) {
    // dx = (1/y) * dout = dout / y
    auto dx_res = out_grad / y;
    if (out_grad.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dx_res, dx);
      } else {
        auto dx_reduce_res =
            dx_res.sum(common::vectorize(reduce_dim),
                       x.dtype(),
                       dx_res.dims().size() == x.dims().size());
        if (dx_reduce_res.dims() != x.dims()) {
          dx_reduce_res =
              reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        }
        set_output<T>(dx_reduce_res, dx);
      }
    } else {
      set_output<T>(dx_res, dx);
    }
  }  // indicate we will compute dx
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
    if (out_grad.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dy_res, dy);
      } else {
        auto dy_reduce_res =
            dy_res.sum(common::vectorize(reduce_dim),
                       y.dtype(),
                       dy_res.dims().size() == y.dims().size());
        if (dy_reduce_res.dims() != y.dims()) {
          dy_reduce_res =
              reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        }
        set_output<T>(dy_reduce_res, dy);
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
    if (out_grad.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dx_res, dx);
      } else {
        auto dx_reduce_res =
            dx_res.sum(common::vectorize(reduce_dim),
                       x.dtype(),
                       dx_res.dims().size() == x.dims().size());
        if (dx_reduce_res.dims() != x.dims()) {
          dx_reduce_res =
              reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        }
        set_output<T>(dx_reduce_res, dx);
      }

    } else {
      set_output<T>(dx_res, dx);
    }
  }  // indicate we will compute dx
}

template <typename T>
void sqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // This calculation is important for resnet.
    auto x_grad_tmp = (0.5 / out) * out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void rsqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // This calculation is important for resnet.
    auto x_grad_tmp = -0.5 * out * out * out * out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void floor_grad(const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto zero_tensor = full<T>(common::vectorize(out_grad.dims()),
                               0.0,
                               out_grad.dtype(),
                               out_grad.place());
    set_output<T>(zero_tensor, x_grad);
  }
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
      split<T>(out_grad, phi::IntArray(sections), axis_value);
  for (int i = 0; i < x_num; ++i) {
    if (x_grad[i]) {
      set_output<T>(x_grad_tmp[i], x_grad[i]);
    }
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
            common::vectorize(axes),
            x_grad_unreduce.dtype(),
            x_grad_unreduce.dims().size() == x.dims().size());
        if (x_grad_reduced.dims() != x.dims()) {
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
            common::vectorize(axes),
            y_grad_unreduce.dtype(),
            y_grad_unreduce.dims().size() != y.dims().size());
        if (y_grad_reduced.dims() != y.dims()) {
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
void expand_grad(const Tensor& x,
                 const Tensor& out_grad,
                 const IntArray& shape,
                 Tensor* x_grad) {
  if (x_grad) {
    auto out_dims = common::make_ddim(shape.GetData());
    if (out_dims != x.dims()) {
      auto axes = get_reduce_dims(x.dims(), out_dims);
      if (!axes.size()) {
        by_pass<T>(out_grad, x_grad);
      } else {
        auto reduced = out_grad.sum(common::vectorize(axes),
                                    x.dtype(),
                                    out_grad.dims().size() == x.dims().size());
        if (reduced.dims() != x.dims()) {
          reduced = reshape<T>(reduced, x.shape());
        }
        set_output<T>(reduced, x_grad);
      }
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
void sigmoid_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    set_output<T>(out_grad * (out * (1 - out)), x_grad);
  }
}

template <typename T>
void abs_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto sign_tmp = sign<T>(x);
    set_output<T>(out_grad * sign_tmp, x_grad);
  }
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
      reshape_out_grad = full<T>({1}, 1, input.dtype(), input.place());
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
  if (data_layout_ != DataLayout::kNCHW) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported storage order: %s", data_layout));
  }
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

  std::vector<int64_t> x_dims = common::vectorize<int64_t>(x.dims());
  auto add_axis = std::vector<int64_t>({-1});
  const int N = x_dims[0];
  const int C = x_dims[1];

  const int hw = x_dims[2] * x_dims[3];
  const int g_num = C / groups;

  auto reduce_axis = IntArray(std::vector<int64_t>({2, 3}));
  auto shape_group = IntArray(std::vector<int64_t>({N, groups, g_num}));
  auto whole_group_shape =
      IntArray(std::vector<int64_t>({N, groups, g_num, hw}));

  auto scale_ptr = scale.get_ptr();
  auto bias_ptr = bias.get_ptr();
  auto inv_std = sqrt<T>(1.0 / (variance + epsilon));
  auto inv_std_mul_s = inv_std / hw / g_num;
  auto dtype = x_data.dtype();
  auto sum_y_grad_mul_x =
      sum<T>(out_grad_data * x_data, reduce_axis, dtype, false);
  auto sum_y_grad = sum<T>(out_grad_data, reduce_axis, dtype, false);
  if (x_grad) {
    Tensor d1;
    Tensor d2;
    Tensor p1;
    if (scale_ptr) {
      auto scale_data = scale.get();
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
      d1 = (reshape<T>(sum_y_grad_mul_x, shape_group))
               .sum(std::vector<int64_t>({2}), dtype, false);
      d2 = (reshape<T>(sum_y_grad, shape_group))
               .sum(std::vector<int64_t>({2}), dtype, false);
      p1 = (reshape<T>(inv_std, std::vector<int64_t>({N, groups, 1})))
               .expand(IntArray(shape_group));
    }

    auto p2 = (d2 * mean - d1) * (inv_std_mul_s * inv_std * inv_std);
    auto p3 = -p2 * mean - d2 * inv_std_mul_s;
    auto first_shape = get_unsqueeze_dims(p1, std::vector<int64_t>({3}));
    auto second_shape = get_unsqueeze_dims(p2, std::vector<int64_t>({2, 3}));
    p1 = reshape<T>(p1, first_shape);
    p2 = reshape<T>(p2, second_shape);
    p3 = reshape<T>(p3, second_shape);
    auto tmp_1 = reshape<T>(out_grad_data, whole_group_shape) * p1;
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
    if (scale_ptr) {
      auto third_shape = get_unsqueeze_dims(mean, std::vector<int64_t>({2}));
      auto tmp1 = (reshape<T>(sum_y_grad_mul_x, shape_group) -
                   reshape<T>(sum_y_grad, shape_group) *
                       reshape<T>(mean, third_shape)) *
                  reshape<T>(inv_std, third_shape);
      auto scale_grad_tmp = reshape<T>(
          tmp1.sum(std::vector<int64_t>({0}), scale_ptr->dtype(), false),
          IntArray(std::vector<int64_t>({C})));
      set_output<T>(scale_grad_tmp, scale_grad);
    }
  }

  if (bias_grad) {
    if (bias_ptr) {
      auto bias_grad_tmp =
          sum_y_grad.sum(std::vector<int64_t>({0}), bias_ptr->dtype(), false);
      set_output<T>(bias_grad_tmp, bias_grad);
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
    }
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
void split_grad(const std::vector<Tensor>& out_grad,
                const Scalar& axis,
                Tensor* x_grad) {
  if (x_grad) {
    auto grad = concat<T>(out_grad, axis);
    set_output<T>(grad, x_grad);
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
    auto zero_tensor =
        full<T>(common::vectorize(x.dims()), 0, x.dtype(), x.place());
    auto x_grad_tmp = put_along_axis<T>(zero_tensor, indices, out_grad, axis);
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void gather_nd_grad(const Tensor& x,
                    const Tensor& index,
                    const Tensor& out_grad,
                    Tensor* x_grad) {
  if (x_grad) {
    auto zero_tensor =
        full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place());
    auto x_grad_tmp = scatter_nd_add<T>(zero_tensor, index, out_grad);
    set_output<T>(x_grad_tmp, x_grad);
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
    auto zero_tensor = full<T>(x_dim, 0.0, x.dtype(), x.place());
    auto zero_mask = cast<T>(equal<T>(x, zero_tensor), x.dtype());
    // determine the index of first zero
    auto zero_mask_cumsum_exclusive =
        cumsum<T>(zero_mask, dim, false, true, reverse);
    auto zero_mask_cumsum = scale<T>(zero_mask_cumsum_exclusive, 2) + zero_mask;
    auto ones_tensor = full<T>(x_dim, 1.0, x.dtype(), x.place());
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
  auto zero_tensor =
      full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place());
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
void min_grad(const Tensor& x,
              const Tensor& out,
              const Tensor& out_grad,
              const IntArray& axis,
              bool keepdim,
              bool reduce_all,
              Tensor* x_grad) {
  if (!x_grad) {
    return;
  }
  auto zero_tensor =
      full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place());
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
void assign_grad(const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    by_pass<T>(out_grad, x_grad);
  }
}

template <typename T>
void erf_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto m_2_sqrt_pi =
        full<T>(common::vectorize(x.dims()), M_2_SQRTPI, x.dtype(), x.place());
    auto neg_one =
        full<T>(common::vectorize(x.dims()), -1.0, x.dtype(), x.place());
    auto neg_tmp = neg_one * x * x;
    auto mul_tmp = m_2_sqrt_pi * exp<T>(neg_tmp);
    set_output<T>(out_grad * mul_tmp, x_grad);
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
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dx_res, x_grad);
      } else {
        auto dx_reduce_res =
            dx_res.sum(common::vectorize(reduce_dim),
                       x.dtype(),
                       dx_res.dims().size() == x.dims().size());
        if (dx_reduce_res.dims() != x.dims()) {
          dx_reduce_res =
              reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        }
        set_output<T>(dx_reduce_res, x_grad);
      }
    } else {
      set_output<T>(dx_res, x_grad);
    }
  }

  if (y_grad) {
    auto y_tmp = cast<T>(less_equal<T>(x, y), out_grad.dtype());
    auto dy_res = out_grad * y_tmp;
    if (out_grad.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dy_res, y_grad);
      } else {
        auto dy_reduce_res =
            dy_res.sum(common::vectorize(reduce_dim),
                       y.dtype(),
                       dy_res.dims().size() == y.dims().size());
        if (dy_reduce_res.dims() != y.dims()) {
          dy_reduce_res =
              reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        }
        set_output<T>(dy_reduce_res, y_grad);
      }
    } else {
      set_output<T>(dy_res, y_grad);
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
void scatter_grad(const Tensor& index,
                  const Tensor& updates,
                  const Tensor& out_grad,
                  bool overwrite,
                  Tensor* x_grad,
                  Tensor* updates_grad) {
  if (x_grad) {
    auto zero_tensor = full<T>(common::vectorize(updates.dims()),
                               0.0,
                               updates.dtype(),
                               updates.place());
    auto tmp_grad = scatter<T>(out_grad, index, zero_tensor, false);
    set_output<T>(tmp_grad, x_grad);
  }

  if (updates_grad) {
    Scalar tmp_zero = 0;
    auto tmp_updates_grad = gather<T>(out_grad, index, tmp_zero);

    // NOTE: len(index) can be smaller than len(updates) when updates is not a
    // scalar
    auto updates_dims = common::vectorize(updates.dims());
    auto index_dims = common::vectorize(index.dims());
    if (updates_dims.size() > 0 && updates_dims[0] > index_dims[0]) {
      // Pad zeros to the end of tmp_updates_grad to make its shape the same as
      // updates.
      decltype(updates_dims) padding_dims = updates_dims;
      padding_dims[0] = updates_dims[0] - index_dims[0];
      auto padding_zeros =
          full<T>(padding_dims, 0, updates.dtype(), updates.place());
      tmp_updates_grad =
          concat<T>({tmp_updates_grad, std::move(padding_zeros)}, 0);
    }
    set_output<T>(tmp_updates_grad, updates_grad);
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
    auto eps = full<T>(common::vectorize(run_var.dims()),
                       epsilon,
                       run_var.dtype(),
                       run_var.place());
    mean_data = run_mean;
    rsqrt_var = (run_var + eps).pow(-0.5);
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
      PADDLE_THROW(common::errors::InvalidArgument("Unknown storage order: %s",
                                                   data_layout));
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
        reshape<T>(scale.get_ptr()
                       ? scale.get()
                       : full<T>(IntArray({c}), 1., x.dtype(), x.place()),
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
void minimum_grad(const Tensor& x,
                  const Tensor& y,
                  const Tensor& out_grad,
                  Tensor* x_grad,
                  Tensor* y_grad) {
  if (x_grad) {
    auto x_tmp = cast<T>(less_than<T>(x, y), out_grad.dtype());
    auto dx_res = out_grad * x_tmp;
    if (out_grad.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dx_res, x_grad);
      } else {
        auto dx_reduce_res =
            dx_res.sum(common::vectorize(reduce_dim),
                       x.dtype(),
                       dx_res.dims().size() == x.dims().size());
        if (dx_reduce_res.dims() != x.dims()) {
          dx_reduce_res =
              reshape<T>(dx_reduce_res, common::vectorize(x.dims()));
        }
        set_output<T>(dx_reduce_res, x_grad);
      }
    } else {
      set_output<T>(dx_res, x_grad);
    }
  }

  if (y_grad) {
    auto y_tmp = cast<T>(greater_equal<T>(x, y), out_grad.dtype());
    auto dy_res = out_grad * y_tmp;
    if (out_grad.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), out_grad.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dy_res, y_grad);
      } else {
        auto dy_reduce_res =
            dy_res.sum(common::vectorize(reduce_dim),
                       y.dtype(),
                       dy_res.dims().size() == y.dims().size());
        if (dy_reduce_res.dims() != y.dims()) {
          dy_reduce_res =
              reshape<T>(dy_reduce_res, common::vectorize(y.dims()));
        }
        set_output<T>(dy_reduce_res, y_grad);
      }
    } else {
      set_output<T>(dy_res, y_grad);
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
      result = full<T>(
          common::vectorize(split_arr[0].dims()), 0.0, x.dtype(), x.place());
      for (int j = 0; j < static_cast<int>(split_arr.size()); j++) {
        result = split_arr[j] + result;
      }
    }
    result = reshape<T>(result, x.shape());
    set_output<T>(result, x_grad);
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
void p_norm_grad(const Tensor& x,
                 /*output of forward was reserved for efficient backward*/
                 const Tensor& out,
                 const Tensor& out_grad,
                 float porder,
                 int axis,
                 float epsilon,
                 bool keepdim,
                 bool asvector,
                 Tensor* x_grad) {
  if (x_grad) {
    if (axis < 0) {
      axis += x.dims().size();
    }

    Tensor x_grad_tmp;
    if (porder == 0.0) {
      // dx = 0
      x_grad_tmp = full<T>(x.shape(), 0, x.dtype(), x.place());
    } else {
      /* generic case formula:
          dx = {
            dy * y^(1-p) * |x|^(p-1) * sgn(x), if p != +-inf,
            dy * sgn(x) * (x==y), if p == +-inf.
          }
      */
      Tensor expand_out = out;
      Tensor expand_out_grad = out_grad;
      // firstly expand output_grad to same ndim with x for convenience
      if (!keepdim) {
        auto expand_shape = common::vectorize(out_grad.dims());
        expand_shape.insert(expand_shape.begin() + axis, 1);
        expand_out_grad = reshape<T>(out_grad, expand_shape);
        expand_out = reshape<T>(out, expand_shape);
      }

      if (porder == 1.0) {
        // dx = dy * sign(x)
        auto x_sign = sign<T>(x);
        x_grad_tmp = x_sign * expand_out_grad;
      } else if (porder == 2.0) {
        // dx = dy * (x / y)
        x_grad_tmp = x / expand_out;
        // fill zero to avoid division by zero
        auto _zero_tensor =
            full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place());
        auto finite_mask = isfinite<T>(x_grad_tmp);
        x_grad_tmp = where<T>(finite_mask, x_grad_tmp, _zero_tensor);
        x_grad_tmp = expand_out_grad * (x_grad_tmp);

      } else if (porder == INFINITY || porder == -INFINITY) {
        // dy * sgn(x) * (x==y), if p == +-inf.
        auto x_abs = abs<T>(x);
        auto mask =
            cast<T>(bitwise_or<T>(equal<T>(x_abs, expand_out), isnan<T>(x_abs)),
                    expand_out.dtype());
        auto x_sign = sign<T>(x);
        x_grad_tmp =
            x_sign * ((expand_out_grad /
                       sum<T>(mask, {axis}, expand_out_grad.dtype(), true)) *
                      mask);

      } else if (porder < 1.0) {
        // dx = dy * y^(1-p) * |x|^(p-1) * sgn(x)
        auto x_sign = sign<T>(x);
        auto x_abs_pow = abs<T>(x);
        x_abs_pow = x_abs_pow.pow(porder - 1);

        auto x_scaled = x_sign * x_abs_pow;
        x_grad_tmp = x_scaled * expand_out_grad * expand_out.pow(1 - porder);

      } else if (porder < 2.0) {
        // dx = dy * y^(1-p) * |x|^(p-1) * sgn(x)
        auto x_sign = sign<T>(x);
        auto x_abs_pow = abs<T>(x);
        x_abs_pow = x_abs_pow.pow(porder - 1);

        // auto scale_v = expand_out_grad / expand_out.pow(porder - 1);
        // auto _zero_tensor =
        //     full<T>(common::vectorize(x.dims()), 0.0, x.dtype());
        // auto out_non_zero_mask = not_equal<T>(expand_out, _zero_tensor);
        // scale_v = scale_v * cast<T>(out_non_zero_mask, scale_v.dtype());
        // x_grad_tmp = x_sign * x_abs_pow * scale_v;

        auto scale_v = expand_out_grad * expand_out.pow(1 - porder);
        x_grad_tmp = x_sign * x_abs_pow * scale_v;

      } else {
        // dx = dy * y^(1-p) * |x|^(p-1) * sgn(x)
        auto x_sign = sign<T>(x);
        auto x_abs_pow = abs<T>(x);
        x_abs_pow = x_abs_pow.pow(porder - 1);

        auto x_scaled = x_sign * x_abs_pow;
        x_grad_tmp = x_scaled * expand_out_grad * expand_out.pow(1 - porder);
      }
    }
    set_output<T>(x_grad_tmp, x_grad);
  }
}

}  // namespace prim
}  // namespace paddle
