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

#include "paddle/fluid/prim/api/all.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace prim {
using Tensor = paddle::Tensor;
using IntArray = paddle::experimental::IntArrayBase<paddle::Tensor>;
//  This file define high level grad composite api for Higher order
//  differentiation

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
void tanh_triple_grad(const Tensor& out,
                      const Tensor& grad_out_forward,
                      const Tensor& grad_x_grad_forward,
                      const paddle::optional<Tensor>& grad_out_new_grad,
                      const paddle::optional<Tensor>& grad_out_grad_grad,
                      Tensor* out_grad,
                      Tensor* grad_out_forward_grad,
                      Tensor* grad_x_grad_forward_grad) {
  if (out_grad) {
    if (grad_out_grad_grad) {
      if (grad_out_new_grad) {
        auto out_grad_tmp =
            (-2 * out * grad_x_grad_forward * grad_out_grad_grad.get()) -
            (2 * grad_out_forward * grad_x_grad_forward *
             grad_out_new_grad.get());
        set_output<T>(out_grad_tmp, out_grad);
      } else {
        auto out_grad_tmp =
            -2 * out * grad_x_grad_forward * grad_out_grad_grad.get();
        set_output<T>(out_grad_tmp, out_grad);
      }
    } else {
      if (grad_out_new_grad) {
        auto out_grad_tmp = -(2 * grad_out_forward * grad_x_grad_forward *
                              grad_out_new_grad.get());
        set_output<T>(out_grad_tmp, out_grad);
      } else {
        auto out_grad_tmp = 0 * out;
        set_output<T>(out_grad_tmp, out_grad);
      }
    }
  }

  if (grad_out_forward_grad) {
    if (grad_out_new_grad) {
      auto grad_out_forward_grad_tmp =
          -2 * out * grad_x_grad_forward * grad_out_new_grad.get();
      set_output<T>(grad_out_forward_grad_tmp, grad_out_forward_grad);
    } else {
      auto grad_out_forward_grad_tmp = 0 * out;
      set_output<T>(grad_out_forward_grad_tmp, grad_out_forward_grad);
    }
  }

  if (grad_x_grad_forward_grad) {
    if (grad_out_grad_grad) {
      if (grad_out_new_grad) {
        auto grad_x_grad_forward_grad_tmp =
            (1 - (out * out)) * grad_out_grad_grad.get() -
            2 * out * grad_out_forward * grad_out_new_grad.get();
        set_output<T>(grad_x_grad_forward_grad_tmp, grad_x_grad_forward_grad);
      } else {
        auto grad_x_grad_forward_grad_tmp =
            (1 - (out * out)) * grad_out_grad_grad.get();
        set_output<T>(grad_x_grad_forward_grad_tmp, grad_x_grad_forward_grad);
      }
    } else {
      if (grad_out_new_grad) {
        auto grad_x_grad_forward_grad_tmp =
            -(2 * out * grad_out_forward * grad_out_new_grad.get());
        set_output<T>(grad_x_grad_forward_grad_tmp, grad_x_grad_forward_grad);
      } else {
        auto grad_x_grad_forward_grad_tmp = 0 * grad_x_grad_forward;
        set_output<T>(grad_x_grad_forward_grad_tmp, grad_x_grad_forward_grad);
      }
    }
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
void silu_double_grad(const Tensor& x,
                      const Tensor& out,
                      const Tensor& out_grad,
                      const Tensor& grad_x_grad,
                      Tensor* grad_x,
                      Tensor* grad_out_grad) {
  auto sigmoid = 1 / (1 + exp<T>(-x));
  auto tmp1 = 1 - sigmoid;
  auto tmp2 = 1 + tmp1 * x;
  if (grad_out_grad) {
    auto ddout = grad_x_grad * sigmoid * tmp2;
    set_output<T>(ddout, grad_out_grad);
  }
  if (grad_x) {
    auto dx = sigmoid * grad_x_grad * out_grad * (1 + (tmp2 - out)) * tmp1;
    set_output<T>(dx, grad_x);
  }
}

template <typename T>
void multiply_double_grad(const Tensor& x,
                          const Tensor& y,
                          const Tensor& grad_out,
                          const paddle::optional<Tensor>& grad_x_grad,
                          const paddle::optional<Tensor>& grad_y_grad,
                          int axis,
                          Tensor* x_grad,
                          Tensor* y_grad,
                          Tensor* grad_out_grad) {
  if (x_grad) {
    if (grad_y_grad) {
      auto dx = grad_y_grad.get() * grad_out;
      if (dx.dims() != x.dims()) {
        auto axes = get_reduce_dims_from_out(dx.dims(), x.dims());
        if (!axes.size()) {
          set_output<T>(dx, x_grad);
        } else {
          auto dx_reduce = dx.sum(phi::vectorize(axes), dx.dtype(), false);
          if (dx_reduce.dims().size() != x.dims().size()) {
            dx_reduce = reshape<T>(dx_reduce, x.shape());
          }
          set_output<T>(dx_reduce, x_grad);
        }
      } else {
        set_output<T>(dx, x_grad);
      }

    } else {
      auto dx = full<T>(phi::vectorize(x.dims()), 0.0, x.dtype());
      set_output<T>(dx, x_grad);
    }
  }
  if (y_grad) {
    if (grad_x_grad) {
      auto dy = grad_x_grad.get() * grad_out;
      if (dy.dims() != y.dims()) {
        auto axes = get_reduce_dims_from_out(dy.dims(), y.dims());
        if (!axes.size()) {
          set_output<T>(dy, y_grad);
        } else {
          auto dy_reduce = dy.sum(phi::vectorize(axes), dy.dtype(), false);
          if (dy_reduce.dims().size() != y.dims().size()) {
            dy_reduce = reshape<T>(dy_reduce, y.shape());
          }
          set_output<T>(dy_reduce, y_grad);
        }
      } else {
        set_output<T>(dy, y_grad);
      }
    } else {
      auto dy = full<T>(phi::vectorize(y.dims()), 0.0, y.dtype());
      set_output<T>(dy, y_grad);
    }
  }
  if (grad_out_grad) {
    Tensor ddout;
    if (grad_x_grad && grad_y_grad) {
      ddout = grad_x_grad.get() * y + grad_y_grad.get() * x;
    } else if (grad_x_grad) {
      ddout = grad_x_grad.get() * y;
    } else if (grad_y_grad) {
      ddout = grad_y_grad.get() * x;
    } else {
      ddout = full<T>(phi::vectorize(grad_out.dims()), 0.0, grad_out.dtype());
    }
    set_output<T>(ddout, grad_out_grad);
  }
}

template <typename T>
void add_double_grad(const Tensor& y,
                     const Tensor& grad_out,
                     const paddle::optional<Tensor>& grad_x_grad,
                     const paddle::optional<Tensor>& grad_y_grad,
                     int axis,
                     Tensor* grad_out_grad) {
  if (grad_out_grad) {
    // ddout = ddx + ddy
    Tensor ddout = full<T>(phi::vectorize(grad_out.dims()), 0.0, y.dtype());
    if (!grad_x_grad && !grad_y_grad) {
      set_output<T>(ddout, grad_out_grad);
    } else {
      if (grad_x_grad) {
        ddout = ddout + grad_x_grad.get();
      }
      if (grad_y_grad) {
        ddout = ddout + grad_y_grad.get();
      }
      set_output<T>(ddout, grad_out_grad);
    }
  }
}

template <typename T>
void add_triple_grad(const paddle::optional<Tensor>& grad_grad_x,
                     const paddle::optional<Tensor>& grad_grad_y,
                     const Tensor& grad_grad_out_grad,
                     int axis,
                     Tensor* grad_grad_x_grad,
                     Tensor* grad_grad_y_grad) {
  if (grad_grad_y_grad) {
    if (grad_grad_y) {
      if (grad_grad_y.get().dims() != grad_grad_out_grad.dims()) {
        // Maybe need reduce here
        phi::DDim reduce_dim = get_reduce_dims(grad_grad_y.get().dims(),
                                               grad_grad_out_grad.dims());
        if (!reduce_dim.size()) {
          by_pass<T>(grad_grad_out_grad, grad_grad_y_grad);
        } else {
          auto dddy_reduce_res = grad_grad_out_grad.sum(
              phi::vectorize(reduce_dim), grad_grad_y.get().dtype(), false);
          auto dddy_tmp = reshape<T>(dddy_reduce_res,
                                     phi::vectorize(grad_grad_y.get().dims()));
          set_output<T>(dddy_tmp, grad_grad_y_grad);
        }
      } else {
        by_pass<T>(grad_grad_out_grad, grad_grad_y_grad);
      }
    } else {
      grad_grad_y_grad = nullptr;
    }
  }
  if (grad_grad_x_grad) {
    if (grad_grad_x) {
      if (grad_grad_x.get().dims() != grad_grad_out_grad.dims()) {
        // Maybe need reduce here
        auto reduce_dim = get_reduce_dims(grad_grad_x.get().dims(),
                                          grad_grad_out_grad.dims());
        if (!reduce_dim.size()) {
          by_pass<T>(grad_grad_out_grad, grad_grad_x_grad);
        } else {
          auto dddx_reduce_res = grad_grad_out_grad.sum(
              phi::vectorize(reduce_dim), grad_grad_x.get().dtype(), false);
          auto dddx_tmp = reshape<T>(dddx_reduce_res,
                                     phi::vectorize(grad_grad_x.get().dims()));
          set_output<T>(dddx_tmp, grad_grad_x_grad);
        }
      } else {
        by_pass<T>(grad_grad_out_grad, grad_grad_x_grad);
      }
    } else {
      grad_grad_x_grad = nullptr;
    }
  }
}

template <typename T>
void subtract_double_grad(const Tensor& y,
                          const Tensor& grad_out,
                          const paddle::optional<Tensor>& grad_x_grad,
                          const paddle::optional<Tensor>& grad_y_grad,
                          int axis,
                          Tensor* grad_out_grad) {
  if (grad_out_grad) {
    // ddout = ddx - ddy
    if (!grad_x_grad && !grad_y_grad) {
      grad_out_grad = nullptr;
    } else {
      Tensor ddout = full<T>(phi::vectorize(grad_out.dims()), 0.0, y.dtype());
      if (grad_x_grad) {
        ddout = ddout + grad_x_grad.get();
      }
      if (grad_y_grad) {
        ddout = ddout - grad_y_grad.get();
      }
      set_output<T>(ddout, grad_out_grad);
    }
  }
}

}  // namespace prim
}  // namespace paddle
