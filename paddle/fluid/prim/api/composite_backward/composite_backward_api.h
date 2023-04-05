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

#include "paddle/fluid/prim/api/all.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace prim {
using Tensor = paddle::Tensor;
using IntArray = paddle::experimental::IntArrayBase<paddle::Tensor>;
//  This function should have as same signature as phi, which defined in
//  paddle/phi/api/backward/backward_api.h
template <typename T>
void relu_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto condition = greater_than<T>(
        out, full<T>(phi::vectorize(out.dims()), 0.0, out.dtype()));
    auto res = where<T>(condition,
                        out_grad,
                        full<T>(phi::vectorize(out.dims()), 0.0, out.dtype()));
    set_output<T>(res, x_grad);
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
      set_output<T>(
          full<T>(phi::vectorize(out_grad.dims()), 0.0, out_grad.dtype()),
          x_grad);
    }
  }
}

template <typename T>
void cast_grad(const Tensor& out_grad, DataType dtype, Tensor* x_grad) {
  if (x_grad) {
    auto res = cast<T>(out_grad, dtype);
    set_output<T>(res, x_grad);
  }
}
template <typename T>
void gather_grad(const Tensor& x,
                 const Tensor& index,
                 const Tensor& out_grad,
                 const Scalar& axis,
                 bool overwrite,
                 Tensor* grad_x) {
  auto zero_tensor = full<T>(phi::vectorize(x.dims()), 0.0, x.dtype());
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
  auto tmp_zero_x_grad = transpose<T>(zero_tensor, tmp_perm);
  auto tmp_out_grad = transpose<T>(out_grad, tmp_perm);
  // scatter grad to grad_x
  auto tmp_grad_x = scatter<T>(tmp_zero_x_grad, index, tmp_out_grad, false);
  auto tmp_grad_x_tranposed = transpose<T>(tmp_grad_x, reverse_perm);
  set_output<T>(tmp_grad_x_tranposed, grad_x);
}

template <typename T>
void tanh_grad(const Tensor& out, const Tensor& grad_out, Tensor* grad_x) {
  if (!grad_x) return;
  auto grad_x_tmp = grad_out * (1 - out * out);
  set_output<T>(grad_x_tmp, grad_x);
}

template <typename T>
void tanh_double_grad(const Tensor& out,
                      const Tensor& grad_out,
                      const Tensor& grad_x_grad,
                      Tensor* out_grad,
                      Tensor* grad_out_grad) {
  // tanh grad grad : ddout = (1 - out^2) * ddx, dout = - (dout_old * 2 * out *
  // ddx)
  auto out_m_grad_x_grad = out * grad_x_grad;
  if (out_grad) {
    auto out_grad_tmp = -2 * grad_out * out_m_grad_x_grad;
    set_output<T>(out_grad_tmp, out_grad);
  }

  if (grad_out_grad) {
    auto grad_out_grad_tmp = grad_x_grad - out * out_m_grad_x_grad;
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
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
        auto dy_reduce_res =
            scale_out_grad.sum(phi::vectorize(reduce_dim), y.dtype(), false);
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
    x_grad_tmp = out_grad.expand(IntArray(x_dim));
  } else {
    if (!keepdim) {
      auto axis_ = std::vector<int64_t>();
      if (reduce_all) {
        for (int64_t i = 1; i < x_dim_size; i++) {
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
      auto out_grad_ = unsqueeze<T>(out_grad, axis_);
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
    // dy = -(x/y^2) * dout
    auto dy_res = -(x / y.pow(2.0)) * out_grad;
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
void elementwise_pow_grad(const Tensor& x,
                          const Tensor& y,
                          const Tensor& out_grad,
                          int axis,
                          Tensor* dx,
                          Tensor* dy) {
  if (dy) {
    // dy = lnx * x^y
    auto lnx = log<T>(x);
    auto x_pow_y = elementwise_pow<T>(x, y);
    auto dy_res = lnx * x_pow_y;
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
    auto dx_res = y * x_pow_z;
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
void sqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // This calculation is important for resnet.
    auto x_grad_tmp = (0.5 / out) * out_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void floor_grad(const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto zero_tensor =
        full<T>(phi::vectorize(out_grad.dims()), 0.0, out_grad.dtype());
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
      split<T>(out_grad, phi::IntArray(sections), axis);
  for (int i = 0; i < x_num; ++i) {
    set_output<T>(x_grad_tmp.at(i), x_grad.at(i));
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
        auto reduced = out_grad.sum(phi::vectorize(axes), x.dtype(), false);
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
void log_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    // dx = dout / x
    set_output<T>(out_grad / x, x_grad);
  }
}

template <typename T>
void exp_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    set_output<T>(out_grad * out, x_grad);
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
    auto abs_tmp = abs<T>(x);
    auto divide_tmp = divide<T>(x, abs_tmp);
    set_output<T>(out_grad * divide_tmp, x_grad);
  }
}

template <typename T>
void matmul_double_grad(const Tensor& x,
                        const Tensor& y,
                        const Tensor& grad_out,
                        const paddle::optional<Tensor>& grad_x_grad,
                        const paddle::optional<Tensor>& grad_y_grad,
                        bool transpose_x,
                        bool transpose_y,
                        Tensor* x_grad,
                        Tensor* y_grad,
                        Tensor* grad_out_grad) {
  // Get dims from the input x, y, output_grad
  std::vector<std::int64_t> x_dims = vectorize(x.dims());
  std::vector<std::int64_t> y_dims = vectorize(y.dims());
  std::vector<std::int64_t> grad_out_dims = vectorize(grad_out.dims());

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int dout_ndim = grad_out_dims.size();

  // prepare dims for x_ndim <= 1 || y_ndim <= 1
  Tensor x_help, y_help, xg_help, yg_help, out_help;

  if (x_ndim == 1 && y_ndim == 1) {
    transpose_x = false;
    transpose_y = false;
    x_help = reshape<T>(x, IntArray(std::vector<int64_t>({1, x_dims[0]})));
    y_help = reshape<T>(y, IntArray(std::vector<int64_t>({y_dims[0], 1})));
    if (grad_x_grad) {
      xg_help = reshape<T>(grad_x_grad.get(),
                           IntArray(std::vector<int64_t>({1, x_dims[0]})));
    }
    if (grad_y_grad) {
      yg_help = reshape<T>(grad_y_grad.get(),
                           IntArray(std::vector<int64_t>({y_dims[0], 1})));
    }
    out_help = reshape<T>(grad_out, IntArray(std::vector<int64_t>({1, 1})));

  } else if (x_ndim == 1) {
    transpose_x = false;
    x_help = reshape<T>(x, IntArray(std::vector<int64_t>({1, x_dims[0]})));
    y_help = y;
    if (grad_x_grad) {
      xg_help = reshape<T>(grad_x_grad.get(),
                           IntArray(std::vector<int64_t>({1, x_dims[0]})));
    }
    if (grad_y_grad) {
      yg_help = grad_y_grad.get();
    }
    auto tmp_grad_out_dims = grad_out_dims;
    tmp_grad_out_dims.insert(tmp_grad_out_dims.begin(), 1);
    out_help = reshape<T>(grad_out, IntArray(tmp_grad_out_dims));

  } else if (y_ndim == 1) {
    transpose_y = false;
    x_help = x;
    y_help = reshape<T>(y, IntArray(std::vector<int64_t>({y_dims[0], 1})));
    if (grad_x_grad) {
      xg_help = grad_x_grad.get();
    }
    if (grad_y_grad) {
      yg_help = reshape<T>(grad_y_grad.get(),
                           IntArray(std::vector<int64_t>({y_dims[0], 1})));
    }
    auto tmp_grad_out_dims = grad_out_dims;
    tmp_grad_out_dims.push_back(1);
    out_help = reshape<T>(grad_out, IntArray(tmp_grad_out_dims));

  } else {
    x_help = x;
    y_help = y;
    if (grad_x_grad) {
      xg_help = grad_x_grad.get();
    }
    if (grad_y_grad) {
      yg_help = grad_y_grad.get();
    }
    out_help = grad_out;
  }

  bool is_broadcast = true;
  if (x_ndim <= 2 && y_ndim <= 2) {
    is_broadcast = false;
  } else if (x_ndim != y_ndim) {
    is_broadcast = true;
  } else {
    is_broadcast = !std::equal(
        x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2, y_dims.cbegin());
  }
  Tensor dx, dy, ddout_1, ddout_2, ddout;
  if (!grad_x_grad && !grad_y_grad) {
    x_grad = nullptr;
    y_grad = nullptr;
    grad_out_grad = nullptr;
    return;

  } else if (!grad_x_grad) {
    y_grad = nullptr;
    if (!transpose_x && !transpose_y) {
      if (x_grad) {
        dx = matmul<T>(out_help, yg_help, false, true);
      }
      if (grad_out_grad) {
        ddout = matmul<T>(x_help, yg_help, false, false);
      }
    } else if (!transpose_x && transpose_y) {
      if (x_grad) {
        dx = matmul<T>(out_help, yg_help, false, false);
      }
      if (grad_out_grad) {
        ddout = matmul<T>(x_help, yg_help, false, true);
      }
    } else if (transpose_x && !transpose_y) {
      if (x_grad) {
        dx = matmul<T>(yg_help, out_help, false, true);
      }
      if (grad_out_grad) {
        ddout = matmul<T>(x_help, yg_help, true, false);
      }
    } else {
      if (x_grad) {
        dx = matmul<T>(yg_help, out_help, true, true);
      }
      if (grad_out_grad) {
        ddout = matmul<T>(x_help, yg_help, true, true);
      }
    }

  } else if (!grad_y_grad) {
    x_grad = nullptr;
    if (!transpose_x && !transpose_y) {
      if (y_grad) {
        dy = matmul<T>(xg_help, out_help, true, false);
      }
      if (grad_out_grad) {
        ddout = matmul<T>(xg_help, y_help, false, false);
      }
    } else if (!transpose_x && transpose_y) {
      if (y_grad) {
        dy = matmul<T>(out_help, xg_help, true, false);
      }
      if (grad_out_grad) {
        ddout = matmul<T>(xg_help, y_help, false, true);
      }
    } else if (transpose_x && !transpose_y) {
      if (y_grad) {
        dy = matmul<T>(xg_help, out_help, false, false);
      }
      if (grad_out_grad) {
        ddout = matmul<T>(xg_help, y_help, true, false);
      }
    } else {
      if (y_grad) {
        dy = matmul<T>(out_help, xg_help, true, true);
      }
      if (grad_out_grad) {
        ddout = matmul<T>(xg_help, y_help, true, true);
      }
    }

  } else {
    if (!transpose_x && !transpose_y) {
      if (x_grad) {
        dx = matmul<T>(out_help, yg_help, false, true);
      }
      if (y_grad) {
        dy = matmul<T>(xg_help, out_help, true, false);
      }
      if (grad_out_grad) {
        ddout_1 = matmul<T>(x_help, yg_help, false, false);
        ddout_2 = matmul<T>(xg_help, y_help, false, false);
        ddout = add<T>(ddout_1, ddout_2);
      }
    } else if (!transpose_x && transpose_y) {
      if (x_grad) {
        dx = matmul<T>(out_help, yg_help, false, false);
      }

      if (y_grad) {
        dy = matmul<T>(out_help, xg_help, true, false);
      }
      if (grad_out_grad) {
        ddout_1 = matmul<T>(x_help, yg_help, false, true);
        ddout_2 = matmul<T>(xg_help, y_help, false, true);
        ddout = add<T>(ddout_1, ddout_2);
      }
    } else if (transpose_x && !transpose_y) {
      if (x_grad) {
        dx = matmul<T>(yg_help, out_help, false, true);
      }

      if (y_grad) {
        dy = matmul<T>(xg_help, out_help, false, false);
      }
      if (grad_out_grad) {
        ddout_1 = matmul<T>(x_help, yg_help, true, false);
        ddout_2 = matmul<T>(xg_help, y_help, true, false);
        ddout = add<T>(ddout_1, ddout_2);
      }
    } else {
      if (x_grad) {
        dx = matmul<T>(yg_help, out_help, true, true);
      }
      if (y_grad) {
        dy = matmul<T>(out_help, xg_help, true, true);
      }
      if (grad_out_grad) {
        ddout_1 = matmul<T>(x_help, yg_help, true, true);
        ddout_2 = matmul<T>(xg_help, y_help, true, true);
        ddout = add<T>(ddout_1, ddout_2);
      }
    }
  }

  if (is_broadcast) {
    // Case3: broadcast. It need cost much time to reduce sum for the
    // broadcast and wastes the memory.
    // So we should avoid the case in reality.
    VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
               "wastes the memory. So we should avoid the case in reality";
    // Reduce sum to get grad by ReduceSum
    if (x_grad) {
      auto tx_dims = x_dims;
      auto tx_ndim = x_ndim;
      auto tdout_ndim = dout_ndim;
      if (x_ndim == 1) {
        tx_dims = std::vector<int64_t>({1, x_dims[0]});
        tx_ndim = x_ndim + 1;
        tdout_ndim = dout_ndim + 1;
      }

      auto x_grad_reduce_dims =
          get_reduce_dims(dx, tdout_ndim, tx_ndim, &tx_dims);

      if (!x_grad_reduce_dims.empty()) {
        dx = sum<T>(dx, IntArray(x_grad_reduce_dims), dy.dtype(), true);
      }
      reshape<T>(dx, IntArray(tx_dims));
    }

    if (y_grad) {
      auto ty_dims = y_dims;
      auto ty_ndim = y_ndim;
      auto tdout_ndim = dout_ndim;
      if (y_ndim == 1) {
        ty_dims = std::vector<int64_t>({y_dims[0], 1});
        ty_ndim = y_ndim + 1;
        tdout_ndim = dout_ndim + 1;
      }

      auto y_grad_reduce_dims =
          get_reduce_dims(dy, tdout_ndim, ty_ndim, &ty_dims);

      if (!y_grad_reduce_dims.empty()) {
        dy = sum<T>(dy, IntArray(y_grad_reduce_dims), dy.dtype(), true);
      }
      reshape<T>(dy, IntArray(ty_dims));
    }
  }

  // recover the original dim of output (delete 1)
  std::vector<int64_t> dx_dims =
      dx.initialized() ? vectorize(dx.dims()) : std::vector<int64_t>({});
  std::vector<int64_t> dy_dims =
      dy.initialized() ? vectorize(dy.dims()) : std::vector<int64_t>({});
  std::vector<int64_t> ddout_dims =
      ddout.initialized() ? vectorize(ddout.dims()) : std::vector<int64_t>({});
  if (x_ndim == 1 && y_ndim == 1) {
    if (dx.initialized() && dx_dims[0] == 1) {
      dx = reshape<T>(dx, IntArray(x_dims));
    }
    if (dy.initialized() && dy_dims.back() == 1) {
      dy = reshape<T>(dy, IntArray(y_dims));
    }
    if (ddout.initialized() && ddout_dims == std::vector<int64_t>({1, 1})) {
      ddout = reshape<T>(ddout, IntArray(std::vector<int64_t>({1})));
    }
  } else if (x_ndim == 1) {
    if (dx.initialized() && dx_dims[0] == 1) {
      dx = reshape<T>(dx, IntArray(x_dims));
    }
    if (ddout.initialized() && ddout_dims[0] == 1) {
      ddout = reshape<T>(ddout,
                         IntArray(std::vector<int64_t>(
                             {ddout_dims.cbegin() + 1, ddout_dims.cend()})));
    }
  } else if (y_ndim == 1) {
    if (dy.initialized() && dy_dims.back() == 1) {
      dy = reshape<T>(dy, IntArray(y_dims));
    }
    if (ddout.initialized() && ddout_dims.back() == 1) {
      ddout = reshape<T>(ddout,
                         IntArray(std::vector<int64_t>(
                             {ddout_dims.cbegin(),
                              ddout_dims.cbegin() + ddout_dims.size() - 1})));
    }
  }

  if (x_grad) {
    set_output<T>(dx, x_grad);
  }
  if (y_grad) {
    set_output<T>(dy, y_grad);
  }
  if (grad_out_grad) {
    set_output<T>(ddout, grad_out_grad);
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
        out_dims = phi::make_ddim(std::vector<int>(decrease_size, 1));
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
        out_dims = phi::make_ddim(origin_out_shape);
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
    if (decrease_size > 0 &&
        (decrease_size != static_cast<size_t>(in_dims.size()))) {
      auto out_tmp =
          pad<T>(reshape<T>(out_grad, origin_out_shape), paddings, 0.0);
      set_output<T>(out_tmp, input_grad);
    } else {
      auto out_tmp = pad<T>(out_grad, paddings, 0.0);
      set_output<T>(out_tmp, input_grad);
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

  // cast dtype to float32 if dtype =float16
  if (x.dtype() == phi::DataType::FLOAT16) {
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
    x_grad_tmp = reshape<T>(x_grad_tmp, phi::vectorize(x.dims()));

    if (x.dtype() == phi::DataType::FLOAT16) {
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
    auto zero_tensor = full<T>(phi::vectorize(x.dims()), 0.0, x.dtype());
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
    auto zero_tensor = full<T>(phi::vectorize(x.dims()), 0.0, x.dtype());
    auto x_grad_tmp = scatter_nd_add<T>(zero_tensor, index, out_grad);
    set_output<T>(x_grad_tmp, x_grad);
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
    auto out_tmp = Tensor();
    if (x_dim_size == 1) {
      x_grad_tmp = out_grad.expand(IntArray(x_dim));
      out_tmp = out.expand(IntArray(x_dim));
    } else {
      if (!keep_dim) {
        auto axis_ = std::vector<int64_t>();
        if (reduce_all) {
          for (int64_t i = 1; i < x_dim_size; i++) {
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
        auto out_grad_ = unsqueeze<T>(out_grad, axis_);
        x_grad_tmp = out_grad_.expand(IntArray(x_dim));
        auto out_ = unsqueeze<T>(out, axis_);
        out_tmp = out_.expand(IntArray(x_dim));
      } else {
        x_grad_tmp = out_grad.expand(IntArray(x_dim));
        out_tmp = out.expand(IntArray(x_dim));
      }
    }
    auto x_grad_res = x_grad_tmp * out_tmp * (1 / x);
    set_output<T>(x_grad_res, x_grad);
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
  auto zero_tensor = full<T>(phi::vectorize(x.dims()), 0.0, x.dtype());
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
  if (x_dim_size == 0 || x_dim_size == 1 || keepdim) {
    auto out_grad_tmp = out_grad.expand(IntArray(x_dim));
    auto out_tmp = out.expand(IntArray(x_dim));
    auto mask = equal<T>(x, out_tmp);
    x_grad_tmp = where<T>(mask, out_grad_tmp, zero_tensor);
  } else {
    auto axis_ = std::vector<int64_t>();
    if (reduce_all) {
      for (int64_t i = 1; i < x_dim_size; i++) {
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
    auto out_grad_ = unsqueeze<T>(out_grad, axis_);
    auto out_ = unsqueeze<T>(out, axis_);
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
    auto m_2_sqrt_pi = full<T>(phi::vectorize(x.dims()), M_2_SQRTPI, x.dtype());
    auto neg_one = full<T>(phi::vectorize(x.dims()), -1.0, x.dtype());
    auto neg_tmp = neg_one * x * x;
    auto mul_tmp = m_2_sqrt_pi * exp<T>(neg_tmp);
    set_output<T>(out_grad * mul_tmp, x_grad);
  }
}

template <typename T>
void maximum_grad(const Tensor& x,
                  const Tensor& y,
                  const Tensor& out_grad,
                  int axis,
                  Tensor* x_grad,
                  Tensor* y_grad) {
  if (x_grad) {
    auto x_tmp = cast<T>(greater_than<T>(x, y), out_grad.dtype());
    auto dx_res = out_grad * x_tmp;
    if (y.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), y.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dx_res, x_grad);
      } else {
        auto dx_reduce_res =
            dx_res.sum(phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, x_grad);
      }
    } else {
      set_output<T>(dx_res, x_grad);
    }
  }

  if (y_grad) {
    auto y_tmp = cast<T>(less_equal<T>(x, y), out_grad.dtype());
    auto dy_res = out_grad * y_tmp;
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dy_res, y_grad);
      } else {
        auto dy_reduce_res =
            dy_res.sum(phi::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
        set_output<T>(dy_tmp, y_grad);
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
    auto zero_tensor =
        full<T>(phi::vectorize(updates.dims()), 0.0, updates.dtype());
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
void batch_norm_grad(const Tensor& x,
                     const Tensor& scale,
                     const Tensor& bias,
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

  DataLayout data_layout_ = phi::StringToDataLayout(data_layout);

  Tensor x_data = x;
  Tensor out_grad_data = out_grad;
  if (x.dtype() == phi::DataType::FLOAT16) {
    x_data = cast<T>(x, phi::DataType::FLOAT32);
  }
  if (out_grad.dtype() == phi::DataType::FLOAT16) {
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
        full<T>(phi::vectorize(run_var.dims()), epsilon, run_var.dtype());
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

      auto x_sub_mean = nhwc_x - mean_data;
      auto sum_dout_mul_diff =
          sum<T>(nhwc_out_grad * x_sub_mean, reduce_axis, dtype, false);

      if (x_grad) {
        if (use_global_stats) {
          auto nhwc_x_grad = scale * rsqrt_var * nhwc_out_grad;
          auto nchw_x_grad = transpose<T>(nhwc_x_grad, nhwc_to_nchw_dim);
          set_output<T>(nchw_x_grad, x_grad);
        } else {
          auto part1 = scale * rsqrt_var;
          auto mean_temp1 = nhwc_out_grad_sum / nhw;
          auto mean_temp2 = sum_dout_mul_diff / nhw * rsqrt_var * rsqrt_var;
          auto part2 = nhwc_out_grad - mean_temp1 - x_sub_mean * mean_temp2;

          auto x_grad_data = part1 * part2;
          auto nchw_x_grad = transpose<T>(x_grad_data, nhwc_to_nchw_dim);
          if (x.dtype() == phi::DataType::FLOAT16) {
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
        set_output<T>(nhwc_out_grad_sum, bias_grad);
      }
      break;
    }
    case DataLayout::kNHWC: {
      if (x_grad) {
        auto x_sub_mean = x_data - mean_data;
        auto out_grad_data_sum =
            sum<T>(out_grad_data, reduce_axis, dtype, false);
        auto nhwc_sum_dout_mul_diff =
            sum<T>(out_grad_data * x_sub_mean, reduce_axis, dtype, false);
        if (use_global_stats) {
          auto x_grad_data = scale * rsqrt_var * out_grad_data;
          set_output<T>(x_grad_data, x_grad);
        } else {
          auto part1 = scale * rsqrt_var;

          auto mean_temp1 = out_grad_data_sum / nhw;
          auto mean_temp2 =
              nhwc_sum_dout_mul_diff / nhw * rsqrt_var * rsqrt_var;
          auto part2 = out_grad_data - mean_temp1 - x_sub_mean * mean_temp2;

          auto x_grad_data = part1 * part2;
          if (x.dtype() == phi::DataType::FLOAT16) {
            x_grad_data = cast<T>(x_grad_data, x.dtype());
          }
          set_output<T>(x_grad_data, x_grad);
        }
        if (scale_grad) {
          auto scale_grad_data = nhwc_sum_dout_mul_diff * rsqrt_var;
          set_output<T>(scale_grad_data, scale_grad);
        }
        if (bias_grad) {
          set_output<T>(out_grad_data_sum, bias_grad);
        }
        break;
      }
    }
    default:
      PADDLE_THROW(phi::errors::InvalidArgument("Unknown storage order: %s",
                                                data_layout));
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

// minimum grad
template <typename T>
void minimum_grad(const Tensor& x,
                  const Tensor& y,
                  const Tensor& out_grad,
                  int axis,
                  Tensor* x_grad,
                  Tensor* y_grad) {
  if (x_grad) {
    auto x_tmp = cast<T>(less_than<T>(x, y), out_grad.dtype());
    auto dx_res = out_grad * x_tmp;
    if (y.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), y.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dx_res, x_grad);
      } else {
        auto dx_reduce_res =
            dx_res.sum(phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, x_grad);
      }
    } else {
      set_output<T>(dx_res, x_grad);
    }
  }

  if (y_grad) {
    auto y_tmp = cast<T>(greater_equal<T>(x, y), out_grad.dtype());
    auto dy_res = out_grad * y_tmp;
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dy_res, y_grad);
      } else {
        auto dy_reduce_res =
            dy_res.sum(phi::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
        set_output<T>(dy_tmp, y_grad);
      }
    } else {
      set_output<T>(dy_res, y_grad);
    }
  }
}

}  // namespace prim
}  // namespace paddle
