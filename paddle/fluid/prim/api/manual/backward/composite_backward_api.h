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
#include "paddle/fluid/prim/api/generated/prim_api/prim_api.h"
#include "paddle/fluid/prim/api/manual/prim_api/prim_api.h"
#include "paddle/fluid/prim/api/manual/utils/utils.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace prim {
using Tensor = paddle::experimental::Tensor;
using IntArray =
    paddle::experimental::IntArrayBase<paddle::experimental::Tensor>;
//  This function should have as same signature as phi, which defined in
//  paddle/phi/api/backward/backward_api.h
template <typename T>
void tanh_grad(const Tensor& out, const Tensor& grad_out, Tensor* grad_x) {
  if (!grad_x) return;
  auto tmp = pow<T>(out, 2.0);
  tmp = scale<T>(tmp, -1.0, 1.0, true);
  auto grad_x_tmp = grad_out * tmp;
  set_output<T>(grad_x_tmp, grad_x);
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
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(scale_out_grad, dy);
      } else {
        auto dy_reduce_res = sum<T>(
            scale_out_grad, phi::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      }
    } else {
      by_pass<T>(scale_out_grad, dy);
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
            sum<T>(out_grad, phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
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
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(out_grad, dy);
      } else {
        auto dy_reduce_res =
            sum<T>(out_grad, phi::vectorize(reduce_dim), y.dtype(), false);
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
            sum<T>(out_grad, phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
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
  std::vector<int> x_dim = phi::vectorize<int>(x.dims());
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
        for (int64_t i = 1; i < x_dim_size; i++) {
          axis_.push_back(i);
        }
      } else {
        axis_ = axis.GetData();
      }
      auto out_grad_ = unsqueeze<T>(out_grad, axis_);
      x_grad_tmp = expand<T>(out_grad_, IntArray(x_dim));
    } else {
      x_grad_tmp = expand<T>(out_grad, IntArray(x_dim));
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
    // dy = -(x/y^2) * dout
    auto tmp0 = pow<T>(y, 2.0);
    auto tmp1 = divide<T>(x, tmp0);
    auto tmp2 = scale<T>(tmp1, -1.0, 0.0, true);
    auto dy_res = tmp2 * out_grad;
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
    auto tmp0 = divide<T>(one_tensor, y);
    auto dx_res = tmp0 * out_grad;
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
void sqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto div_x = full<T>(phi::vectorize(out.dims()), 0.5);
    auto tmp = divide<T>(div_x, out);
    auto x_grad_tmp = out_grad * tmp;
    set_output<T>(x_grad_tmp, x_grad);
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
    if (x.dims() != y.dims()) {
      auto axes = get_reduce_dims(x.dims(), y.dims());
      if (!axes.size()) {
        set_output<T>(x_grad_unreduce, x_grad);
      } else {
        auto x_grad_reduced = sum<T>(x_grad_unreduce,
                                     phi::vectorize(axes),
                                     x_grad_unreduce.dtype(),
                                     false);
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
    auto y_grad_unreduce = multiply<T>(out_grad, x);
    if (y.dims() != x.dims()) {
      auto axes = get_reduce_dims(y.dims(), x.dims());
      if (!axes.size()) {
        set_output<T>(y_grad_unreduce, y_grad);
      } else {
        auto y_grad_reduced = sum<T>(y_grad_unreduce,
                                     phi::vectorize(axes),
                                     y_grad_unreduce.dtype(),
                                     false);
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
void expand_grad(const Tensor& x,
                 const Tensor& out_grad,
                 const IntArray& shape,
                 Tensor* x_grad) {
  if (x_grad) {
    auto out_dims = phi::make_ddim(shape.GetData());
    if (out_dims != x.dims()) {
      auto axes = get_reduce_dims(x.dims(), out_dims);
      if (!axes.size()) {
        by_pass<T>(out_grad, x_grad);
      } else {
        auto reduced = sum<T>(out_grad, phi::vectorize(axes), x.dtype(), false);
        if (reduced.dims().size() != x.dims().size()) {
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
void exp_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    set_output<T>(out_grad * out, x_grad);
  }
}

}  // namespace prim
}  // namespace paddle
