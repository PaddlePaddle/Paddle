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

#include "paddle/fluid/prim/api/all.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
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
    // dy = -(x/y^2) * grad_out
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
    // dx = (1/y) * grad_out
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
    if (x_grad_unreduce.dims() != x.dims()) {
      auto axes = get_reduce_dims_from_out(x_grad_unreduce.dims(), x.dims());
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
    auto y_grad_unreduce = out_grad * x;
    if (y_grad_unreduce.dims() != y.dims()) {
      auto axes = get_reduce_dims_from_out(y_grad_unreduce.dims(), y.dims());
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
  VLOG(1) << "-------------------1----------------";
  std::vector<std::int64_t> x_dims = vectorize(x.dims());
  std::vector<std::int64_t> y_dims = vectorize(y.dims());
  std::vector<std::int64_t> grad_out_dims = vectorize(grad_out.dims());

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int dout_ndim = grad_out_dims.size();

  // prepare dims for x_ndim <= 1 || x_ndim <= 1
  Tensor x_help, y_help, xg_help, yg_help, out_help;
  std::cout << "x_ndim = " << x_ndim << std::endl;
  std::cout << "y_ndim = " << y_ndim << std::endl;
  std::cout << "dout_ndim = " << dout_ndim << std::endl;

  if (x_ndim == 1 && y_ndim == 1) {
    VLOG(1) << "---------x_ndim=1 && y_ndim=1---------";
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
    VLOG(1) << "----------x_ndim=1 && y_ndim>1--------";
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
    VLOG(1) << "----------x_ndim>1 && y_ndim=1--------";
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
    VLOG(1) << "-----------x_ndim>1 && y_ndim>1--------";
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

  std::cout << "xg_help.dims :" << xg_help.dims() << std::endl;
  std::cout << "yg_help.dims :" << yg_help.dims() << std::endl;
  std::cout << "out_help.dims :" << out_help.dims() << std::endl;

  VLOG(1) << "------------broadcast-----------";
  bool is_broadcast = true;
  if (x_ndim <= 2 && y_ndim <= 2) {
    VLOG(1) << "------------broadcast_1 Fasle-----------";
    is_broadcast = false;
  } else if (x_ndim != y_ndim) {
    VLOG(1) << "------------broadcast_2 True-----------";
    is_broadcast = true;
  } else {
    VLOG(1) << "------------broadcast_3 -----------";
    is_broadcast = !std::equal(
        x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2, y_dims.cbegin());
    std::cout << "broadcast = " << is_broadcast << std::endl;
  }
  VLOG(1) << "-------------------4----------------";
  Tensor dx, dy, ddout_1, ddout_2, ddout;
  if (!grad_x_grad && !grad_y_grad) {
    VLOG(1) << "-------------------!ddx!ddy----------------";
    x_grad = nullptr;
    y_grad = nullptr;
    grad_out_grad = nullptr;
    return;

  } else if (!grad_x_grad) {
    VLOG(1) << "-------------------!ddx----------------";
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
    VLOG(1) << "-------------------!ddy----------------";
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
    VLOG(1) << "-------------------ddxddy----------------";
    if (!transpose_x && !transpose_y) {
      if (x_grad) {
        dx = matmul<T>(out_help, yg_help, false, true);
        std::cout << "after matmul dx.dims = " << dx.dims() << std::endl;
      }
      if (y_grad) {
        dy = matmul<T>(xg_help, out_help, true, false);
        std::cout << "after matmul dy.dims = " << dy.dims() << std::endl;
      }
      if (grad_out_grad) {
        ddout_1 = matmul<T>(x_help, yg_help, false, false);
        ddout_2 = matmul<T>(xg_help, y_help, false, false);
        ddout = add<T>(ddout_1, ddout_2);
        std::cout << "after matmul ddout.dims = " << ddout.dims() << std::endl;
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
  VLOG(1) << "------------------5--------------------";

  if (is_broadcast) {
    // Case3: broadcast. It need cost much time to reduce sum for the
    // broadcast and wastes the memory.
    // So we should avoid the case in reality.
    VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
               "wastes the memory. So we should avoid the case in reality";
    VLOG(1) << "-------------------is_broadcast----------------";
    // Reduce sum to get grad by ReduceSum
    if (x_grad) {
      VLOG(1) << "------------------set x_grad---------------";
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

      std::cout << "dx.dims = " << dx.dims() << std::endl;
      if (!x_grad_reduce_dims.empty()) {
        dx = sum<T>(dx, IntArray(x_grad_reduce_dims), dy.dtype(), true);
      }
      reshape<T>(dx, IntArray(tx_dims));
    }
    std::cout << "origin dy.dim " << dy.dims() << std::endl;
    if (y_grad) {
      VLOG(1) << "-------------------set y_grad----------------";
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
      std::cout << "ty.dim should be [2, 1]" << ty_dims[0] << ty_dims[1]
                << std::endl;
      std::cout << "y.dims()" << y.dims() << std::endl;
      reshape<T>(dy, IntArray(ty_dims));
    }
    std::cout << "after reduce dy.dim " << dy.dims() << std::endl;
  }

  // recover the original dim of output (delete 1)
  std::vector<int64_t> dx_dims =
      dx.initialized() ? vectorize(dx.dims()) : std::vector<int64_t>({});
  std::vector<int64_t> dy_dims =
      dy.initialized() ? vectorize(dy.dims()) : std::vector<int64_t>({});
  std::vector<int64_t> ddout_dims =
      ddout.initialized() ? vectorize(ddout.dims()) : std::vector<int64_t>({});
  if (x_ndim == 1 && y_ndim == 1) {
    VLOG(1) << "---------recover x_ndim=1 && y_ndim=1---------";
    if (dx.initialized() && dx_dims[0] == 1) {
      dx = reshape<T>(dx, IntArray(x_dims));
    }
    std::cout << "after recover dx.dims = " << dx.dims() << std::endl;
    if (dy.initialized() && dy_dims.back() == 1) {
      dy = reshape<T>(dy, IntArray(y_dims));
    }
    if (ddout.initialized() && ddout_dims == std::vector<int64_t>({1, 1})) {
      ddout = reshape<T>(ddout, IntArray(std::vector<int64_t>({1})));
    }
  } else if (x_ndim == 1) {
    VLOG(1) << "---------recover x_ndim=1 ---------";
    if (dx.initialized() && dx_dims[0] == 1) {
      dx = reshape<T>(dx, IntArray(x_dims));
      std::cout << "after recover dx.dims = " << dx.dims() << std::endl;
    }
    std::cout << "after recover dx.dims = " << dx.dims() << std::endl;
    if (ddout.initialized() && ddout_dims[0] == 1) {
      ddout = reshape<T>(ddout,
                         IntArray(std::vector<int64_t>(
                             {ddout_dims.cbegin() + 1, ddout_dims.cend()})));
    }
  } else if (y_ndim == 1) {
    VLOG(1) << "---------recover  y_ndim=1---------";
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
    VLOG(1) << "-------------------set x_grad----------------";
    set_output<T>(dx, x_grad);
    std::cout << "set x_grad dx.dims = " << x_grad->dims() << std::endl;
  }
  if (y_grad) {
    VLOG(1) << "-------------------set y_grad----------------";
    set_output<T>(dy, y_grad);
    std::cout << "end y_grad.dim " << y_grad->dims() << std::endl;
  }
  if (grad_out_grad) {
    VLOG(1) << "-------------------set grad_out_grad----------------";
    set_output<T>(ddout, grad_out_grad);
  }
}

}  // namespace prim
}  // namespace paddle
