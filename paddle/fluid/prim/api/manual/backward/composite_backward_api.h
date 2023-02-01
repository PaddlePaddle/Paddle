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
  auto grad_x_tmp = multiply<T>(grad_out, tmp);
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
    // dy = -(x/y^2) * grad_out
    auto tmp0 = pow<T>(y, 2.0);
    auto tmp1 = divide<T>(x, tmp0);
    auto tmp2 = scale<T>(tmp1, -1.0, 0.0, true);
    auto dy_res = multiply<T>(tmp2, out_grad);
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
    // dx = (1/y) * grad_out
    auto one_tensor = full<T>(phi::vectorize(y.dims()), 1.0, y.dtype());
    auto tmp0 = divide<T>(one_tensor, y);
    auto dx_res = multiply<T>(tmp0, out_grad);
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
    auto x_grad_tmp = multiply<T>(out_grad, tmp);
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
    auto x_grad_unreduce = multiply<T>(out_grad, y);
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
    set_output<T>(multiply<T>(out_grad, out), x_grad);
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
  int ndim = grad_out_dims.size();

  // Case1 : x's or y's dim = 1

  bool is_broadcast = true;
  if (x_ndim <= 2 || y_ndim <= 2) {
    is_broadcast = false;
  } else if (x_ndim != y_ndim) {
    is_broadcast = true;
  } else {
    is_broadcast = !std::equal(
        x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2, y_dims.cbegin());
  }

  if (!is_broadcast) {
    // Case2: no broadcast or no batch size
    Tensor x_help = x;
    Tensor y_help = y;
    Tensor grad_out_help = grad_out;

    reshape_xyout_to_matrixsequence<T>(
        x_help, y_help, grad_out_help, transpose_x, transpose_y);

    phi::DDim x_grad_dims;
    if (x_grad) {
      x_grad_dims = x_grad->dims();
      if (x_grad_dims != x_help.dims()) {
        *x_grad = reshape<T>(*x_grad, IntArray(phi::vectorize(x_help.dims())));
      }
    }

    phi::DDim y_grad_dims;
    if (y_grad) {
      y_grad_dims = y_grad->dims();
      if (y_grad_dims != y_help.dims()) {
        *y_grad = reshape<T>(*y_grad, IntArray(phi::vectorize(y_help.dims())));
      }
    }

    phi::DDim dgrad_out_dims;
    if (grad_out_grad) {
      dgrad_out_dims = grad_out_grad->dims();
      if (dgrad_out_dims != grad_out_help.dims()) {
        *grad_out_grad = reshape<T>(
            *grad_out_grad, IntArray(phi::vectorize(grad_out_help.dims())));
      }
    }

    bool dgrad_out_flag = false;
    if (grad_x_grad) {
      auto grad_x_grad_mat = grad_x_grad.get();
      if (grad_x_grad_mat.dims() != x_help.dims()) {
        grad_x_grad_mat = reshape<T>(grad_x_grad_mat,
                                     IntArray(phi::vectorize(x_help.dims())));
      }
      if (y_grad) {
        Tensor y_grad_tmp;
        if (transpose_x && transpose_y) {
          // y_grad = grad_out' * grad_x_grad'
          auto tmp =
              modify_dim_for_matmul<T>(grad_out, true, grad_x_grad_mat, false);
          y_grad_tmp =
              matmul<T>(std::get<0>(tmp), std::get<1>(tmp), true, true);
        } else if (transpose_x) {
          // y_grad = grad_x_grad * grad_out
          auto tmp =
              modify_dim_for_matmul<T>(grad_x_grad_mat, false, grad_out, true);
          y_grad_tmp =
              matmul<T>(std::get<0>(tmp), std::get<1>(tmp), false, false);
        } else if (transpose_y) {
          // y_grad = grad_out' * grad_x_grad
          auto tmp =
              modify_dim_for_matmul<T>(grad_out, true, grad_x_grad_mat, true);
          y_grad_tmp =
              matmul<T>(std::get<0>(tmp), std::get<1>(tmp), true, false);
        } else {
          // y_grad = grad_x_grad' * grad_out
          auto tmp =
              modify_dim_for_matmul<T>(grad_x_grad_mat, true, grad_out, true);
          y_grad_tmp =
              matmul<T>(std::get<0>(tmp), std::get<1>(tmp), true, false);
        }
        set_output<T>(y_grad_tmp, y_grad);
      }

      if (grad_out_grad) {
        auto tmp = modify_dim_for_matmul<T>(grad_x_grad_mat, true, y, false);
        auto grad_out_grad_tmp = matmul<T>(
            std::get<0>(tmp), std::get<1>(tmp), transpose_x, transpose_y);
        set_output<T>(grad_out_grad_tmp, grad_out_grad);
      }
    } else if (!grad_x_grad && y_grad) {
      auto y_grad_tmp = full<T>(phi::vectorize(y.dims()), Scalar(0.0));
      set_output<T>(y_grad_tmp, y_grad);
    }
    if (grad_y_grad) {
      auto grad_y_grad_mat = grad_y_grad.get();
      if (grad_y_grad_mat.dims() != y_help.dims()) {
        grad_y_grad_mat = reshape<T>(grad_y_grad_mat,
                                     IntArray(phi::vectorize(y_help.dims())));
      }
      if (x_grad) {
        Tensor x_grad_tmp;
        if (transpose_x && transpose_y) {
          // x_grad = grad_y_grad' * grad_out'
          auto tmp =
              modify_dim_for_matmul<T>(grad_y_grad_mat, true, grad_out, false);
          x_grad_tmp =
              matmul<T>(std::get<0>(tmp), std::get<1>(tmp), true, true);
        } else if (transpose_x) {
          // x_grad = grad_y_grad * grad_out'
          auto tmp =
              modify_dim_for_matmul<T>(grad_y_grad_mat, false, grad_out, false);
          x_grad_tmp =
              matmul<T>(std::get<0>(tmp), std::get<1>(tmp), false, true);
        } else if (transpose_y) {
          // x_grad = grad_out * grad_y_grad
          auto tmp =
              modify_dim_for_matmul<T>(grad_out, false, grad_y_grad_mat, true);
          x_grad_tmp =
              matmul<T>(std::get<0>(tmp), std::get<1>(tmp), false, false);
        } else {
          // x_grad = grad_out * grad_y_grad'
          auto tmp =
              modify_dim_for_matmul<T>(grad_out, false, grad_y_grad_mat, false);
          x_grad_tmp =
              matmul<T>(std::get<0>(tmp), std::get<1>(tmp), false, true);
        }
        set_output<T>(x_grad_tmp, x_grad);
      }

      if (grad_out_grad) {
        auto tmp = modify_dim_for_matmul<T>(x, true, grad_y_grad_mat, false);
        auto grad_out_grad_tmp = matmul<T>(
            std::get<0>(tmp), std::get<1>(tmp), transpose_x, transpose_y);
        auto output_tmp = add<T>(grad_out_grad_tmp, *grad_out_grad);
        set_output<T>(output_tmp, grad_out_grad);
      }
    } else if (!grad_y_grad && x_grad) {
      auto x_grad_tmp = full<T>(phi::vectorize(x.dims()), Scalar(0.0));
      set_output<T>(x_grad_tmp, x_grad);
    }
    if (grad_out_grad && !grad_x_grad && !grad_y_grad) {
      auto grad_out_grad_tmp =
          full<T>(phi::vectorize(grad_out.dims()), Scalar(0.0));
      set_output<T>(grad_out_grad_tmp, grad_out_grad);
    }

    if (x_grad) {
      if (x_grad_dims != x_help.dims()) {
        *x_grad = reshape<T>(*x_grad, IntArray(phi::vectorize(x_grad_dims)));
      }
    }

    if (y_grad) {
      if (y_grad_dims != y_help.dims()) {
        *y_grad = reshape<T>(*y_grad, IntArray(phi::vectorize(y_grad_dims)));
      }
    }

    if (grad_out_grad) {
      if (dgrad_out_dims != grad_out_help.dims()) {
        *grad_out_grad = reshape<T>(*grad_out_grad,
                                    IntArray(phi::vectorize(dgrad_out_dims)));
      }
    }

  } else {
    // Case3: broadcast. It need cost much time to reduce sum for the
    // broadcast and wastes the memory.
    // So we should avoid the case in reality.
    VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
               "wastes the memory. So we should avoid the case in reality";

    Tensor x_grad_help;
    Tensor y_grad_help;
    Tensor grad_out_grad_help;

    if (transpose_x) {
      if (transpose_y) {
        if (x_grad && grad_y_grad) {
          x_grad_help = matmul<T>(grad_y_grad.get(), grad_out, true, true);
        }
        if (y_grad && grad_x_grad) {
          y_grad_help = matmul<T>(grad_out, grad_x_grad.get(), true, true);
        }
      } else {
        if (x_grad && grad_y_grad) {
          x_grad_help = matmul<T>(grad_y_grad.get(), grad_out, false, true);
        }
        if (y_grad && grad_x_grad) {
          y_grad_help = matmul<T>(grad_x_grad.get(), grad_out, false, false);
        }
      }
    } else {
      if (transpose_y) {
        if (x_grad && grad_y_grad) {
          x_grad_help = matmul<T>(grad_out, grad_y_grad.get(), false, false);
        }
        if (y_grad && grad_x_grad) {
          y_grad_help = matmul<T>(grad_out, grad_x_grad.get(), true, false);
        }
      } else {
        if (x_grad && grad_y_grad) {
          x_grad_help = matmul<T>(grad_out, grad_y_grad.get(), false, true);
        }
        if (y_grad && grad_x_grad) {
          y_grad_help = matmul<T>(grad_x_grad.get(), grad_out, true, false);
        }
      }
    }

    // get help dims
    const std::vector<std::int64_t> x_grad_help_dims =
        vectorize(x_grad_help.dims());
    const std::vector<std::int64_t> y_grad_help_dims =
        vectorize(y_grad_help.dims());

    std::vector<std::int64_t> x_grad_broadcast_dims(ndim);
    std::vector<std::int64_t> y_grad_broadcast_dims(ndim);

    std::fill(x_grad_broadcast_dims.data(),
              x_grad_broadcast_dims.data() + ndim - x_ndim,
              1);
    std::fill(y_grad_broadcast_dims.data(),
              y_grad_broadcast_dims.data() + ndim - y_ndim,
              1);
    std::copy(x_dims.data(),
              x_dims.data() + x_ndim,
              x_grad_broadcast_dims.data() + ndim - x_ndim);
    std::copy(y_dims.data(),
              y_dims.data() + y_ndim,
              y_grad_broadcast_dims.data() + ndim - y_ndim);

    std::vector<int> x_grad_reduce_dims;
    std::vector<int> y_grad_reduce_dims;
    for (int ix_grad = 0; ix_grad <= ndim - 3; ix_grad++) {
      if (x_grad_help_dims[ix_grad] != 1 &&
          x_grad_broadcast_dims[ix_grad] == 1) {
        x_grad_reduce_dims.push_back(ix_grad);
      }
      if (y_grad_help_dims[ix_grad] != 1 &&
          y_grad_broadcast_dims[ix_grad] == 1) {
        y_grad_reduce_dims.push_back(ix_grad);
      }
    }
    // Reduce sum to get grad by ReduceSum
    if (x_grad && x_grad_help.initialized()) {
      if (x_grad_reduce_dims.empty()) {
        x_grad_help = std::move(x_grad_help);
      } else {
        x_grad_help = sum<T>(x_grad_help, IntArray(x_grad_reduce_dims));
      }
      reshape<T>(x_grad_help, IntArray(phi::vectorize(x.dims())));
    } else if (x_grad && !x_grad_help.initialized()) {
      x_grad_help = full<T>(phi::vectorize(x.dims()), Scalar(0.0));
    }
    set_output<T>(x_grad_help, x_grad);

    if (y_grad && y_grad_help.initialized()) {
      if (y_grad_reduce_dims.empty()) {
        y_grad_help = std::move(y_grad_help);
      } else {
        y_grad_help = sum<T>(y_grad_help, IntArray(y_grad_reduce_dims));
      }
      reshape<T>(y_grad_help, IntArray(phi::vectorize(y.dims())));
    } else if (y_grad && !y_grad_help.initialized()) {
      y_grad_help = full<T>(phi::vectorize(y.dims()), Scalar(0.0));
    }
    set_output<T>(y_grad_help, y_grad);

    if (grad_out_grad) {
      // Calculate the gradient of OutputGrad(Out)
      if (grad_x_grad) {
        grad_out_grad_help =
            matmul<T>(grad_x_grad.get(), y, transpose_x, transpose_y);
      }
      if (grad_y_grad) {
        auto grad_out_grad_help_2 =
            matmul<T>(x, grad_y_grad.get(), transpose_x, transpose_y);
        grad_out_grad_help = add<T>(grad_out_grad_help, grad_out_grad_help_2);
      }
      set_output<T>(grad_out_grad_help, grad_out_grad);
    }
  }
}

}  // namespace prim
}  // namespace paddle
