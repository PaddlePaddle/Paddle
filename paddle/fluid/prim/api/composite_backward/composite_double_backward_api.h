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

#include "paddle/common/ddim.h"
#include "paddle/fluid/prim/api/all.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/int_array.h"

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
void sin_double_grad(const Tensor& x,
                     const Tensor& grad_out,
                     const Tensor& grad_x_grad,
                     Tensor* x_grad,
                     Tensor* grad_out_grad) {
  // sin grad grad : ddout = cosx * ddx, dx = -dy * sinx * ddx
  if (x_grad) {
    auto x_grad_tmp = -(grad_out * sin<T>(x) * grad_x_grad);
    set_output<T>(x_grad_tmp, x_grad);
  }

  if (grad_out_grad) {
    auto grad_out_grad_tmp = cos<T>(x) * grad_x_grad;
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }
}

template <typename T>
void cos_double_grad(const Tensor& x,
                     const Tensor& grad_out,
                     const Tensor& grad_x_grad,
                     Tensor* x_grad,
                     Tensor* grad_out_grad) {
  // cos grad grad : ddout = -sinx * ddx, dx = -dy * cosx * ddx
  if (x_grad) {
    auto x_grad_tmp = -(grad_out * cos<T>(x) * grad_x_grad);
    set_output<T>(x_grad_tmp, x_grad);
  }

  if (grad_out_grad) {
    auto grad_out_grad_tmp = -sin<T>(x) * grad_x_grad;
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }
}

template <typename T>
void acos_double_grad(const Tensor& x,
                      const Tensor& grad_out,
                      const Tensor& grad_x_grad,
                      Tensor* x_grad,
                      Tensor* grad_out_grad) {
  // acos grad grad : ddout = -((1-x*x)^(-0.5)) * ddx
  // dx = dy * (-x)*((1-x*x)^(-1.5)) * ddx
  auto x_tmp = 1 - x * x;
  if (x_grad) {
    auto x_grad_tmp = (grad_out * (-x) * pow<T>(x_tmp, -1.5) * grad_x_grad);
    set_output<T>(x_grad_tmp, x_grad);
  }

  if (grad_out_grad) {
    auto grad_out_grad_tmp = -pow<T>(x_tmp, -0.5) * grad_x_grad;
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }
}

template <typename T>
void minimum_double_grad(const Tensor& x,
                         const Tensor& y,
                         const paddle::optional<Tensor>& grad_x_grad,
                         const paddle::optional<Tensor>& grad_y_grad,
                         Tensor* grad_out_grad) {
  if (grad_out_grad) {
    if (grad_x_grad && grad_y_grad) {
      auto x_mask = cast<T>(less_than<T>(x, y), grad_x_grad.get().dtype());
      auto ddout =
          grad_x_grad.get() * x_mask + grad_y_grad.get() * (1 - x_mask);
      set_output<T>(ddout, grad_out_grad);
    } else if (grad_x_grad) {
      auto x_mask = cast<T>(less_than<T>(x, y), grad_x_grad.get().dtype());
      auto ddout = grad_x_grad.get() * x_mask;
      set_output<T>(ddout, grad_out_grad);
    } else if (grad_y_grad) {
      auto y_mask = cast<T>(greater_equal<T>(x, y), grad_y_grad.get().dtype());
      auto ddout = grad_y_grad.get() * y_mask;
      set_output<T>(ddout, grad_out_grad);
    }
  }
}
template <typename T>
void pow_double_grad(const Tensor& x,
                     const Tensor& grad_out,
                     const Tensor& grad_x_grad,
                     const Scalar& y,
                     Tensor* x_grad,
                     Tensor* grad_out_grad) {
  // pow grad grad : ddout = y * pow(x, y-1) * ddx, dx = y * (y-1) * pow(x, y-2)
  // * dout * ddx
  auto y_value = y.to<float>();
  if (grad_out_grad) {
    auto grad_out_grad_tmp = y_value * x.pow(y_value - 1) * grad_x_grad;
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }

  if (x_grad) {
    auto x_grad_tmp =
        y_value * (y_value - 1) * x.pow(y_value - 2) * grad_out * grad_x_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void maximum_double_grad(const Tensor& x,
                         const Tensor& y,
                         const paddle::optional<Tensor>& grad_x_grad,
                         const paddle::optional<Tensor>& grad_y_grad,
                         Tensor* grad_out_grad) {
  if (grad_out_grad) {
    if (grad_x_grad && grad_y_grad) {
      auto x_mask = cast<T>(greater_than<T>(x, y), grad_x_grad.get().dtype());
      auto ddout =
          grad_x_grad.get() * x_mask + grad_y_grad.get() * (1 - x_mask);
      set_output<T>(ddout, grad_out_grad);
    } else if (grad_x_grad) {
      auto x_mask = cast<T>(greater_than<T>(x, y), grad_x_grad.get().dtype());
      auto ddout = grad_x_grad.get() * x_mask;
      set_output<T>(ddout, grad_out_grad);
    } else if (grad_y_grad) {
      auto y_mask = cast<T>(less_equal<T>(x, y), grad_y_grad.get().dtype());
      auto ddout = grad_y_grad.get() * y_mask;
      set_output<T>(ddout, grad_out_grad);
    }
  }
}

template <typename T>
void where_double_grad(const Tensor& condition,
                       const paddle::optional<Tensor>& grad_x_grad,
                       const paddle::optional<Tensor>& grad_y_grad,
                       Tensor* grad_out_grad) {
  if (grad_out_grad) {
    Tensor ddout;
    if (grad_x_grad && grad_y_grad) {
      // ddz = ddx * cond + (1-cond) * ddy
      ddout = where<T>(condition, grad_x_grad.get(), grad_y_grad.get());
    } else if (grad_x_grad) {
      // ddz = ddx * cond
      auto zeros = full<T>(common::vectorize(grad_x_grad.get().dims()),
                           0,
                           grad_x_grad.get().dtype(),
                           grad_x_grad.get().place());
      ddout = where<T>(condition, grad_x_grad.get(), zeros);
    } else if (grad_y_grad) {
      // ddz = (1 - cond) * ddy
      auto zeros = full<T>(common::vectorize(grad_y_grad.get().dims()),
                           0,
                           grad_y_grad.get().dtype(),
                           grad_y_grad.get().place());
      ddout = where<T>(condition, zeros, grad_y_grad.get());
    }
    set_output<T>(ddout, grad_out_grad);
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
  if (grad_out_new_grad && grad_out_grad_grad) {
    /*
    dy = -2 * dy * ddx * ddy - 2 * y * ddx * dddy
    ddy = -2 * y * ddx * ddy
    dddx = -2 * y * dy * ddy + (1 - y^2) * dddy
    */
    /* precompute '-2 * y' to prevent duplicated computation*/
    Tensor neg_2_out;
    if (grad_out_forward_grad || grad_x_grad_forward_grad) {
      neg_2_out = scale<T>(out, -2.0);
    }
    /* precompute 'dy(prev) * ddy' to prevent duplicated computation*/
    Tensor grad_out_forward_mul_grad_out_new_grad;
    if (out_grad || grad_x_grad_forward_grad) {
      grad_out_forward_mul_grad_out_new_grad =
          grad_out_forward * grad_out_new_grad.get();
    }

    if (out_grad) {
      auto out_grad_tmp = (scale<T>(grad_x_grad_forward, -2.0) *
                           (grad_out_forward_mul_grad_out_new_grad +
                            out * grad_out_grad_grad.get()));
      set_output<T>(out_grad_tmp, out_grad);
    }
    if (grad_out_forward_grad) {
      auto grad_out_forward_grad_tmp =
          (neg_2_out * grad_x_grad_forward * grad_out_new_grad.get());
      set_output<T>(grad_out_forward_grad_tmp, grad_out_forward_grad);
    }
    if (grad_x_grad_forward_grad) {
      auto grad_x_grad_forward_grad_tmp =
          (scale<T>(out * out, -1.0, 1.0) * grad_out_grad_grad.get() +
           neg_2_out * grad_out_forward_mul_grad_out_new_grad);
      set_output<T>(grad_x_grad_forward_grad_tmp, grad_x_grad_forward_grad);
    }

  } else if (grad_out_new_grad) {
    /*
    dy = -2 * dy * ddx * ddy
    ddy = -2 * y * ddx * ddy
    dddx = -2 * y * dy * ddy
    */
    // regard 'grad_out_grad_grad' is zero
    /* precompute '-2 * y' to prevent duplicated computation*/
    Tensor neg_2_out;
    if (grad_out_forward_grad || grad_x_grad_forward_grad) {
      neg_2_out = scale<T>(out, -2.0);
    }
    /* precompute 'dy(prev) * ddy' to prevent duplicated computation*/
    Tensor grad_out_forward_mul_grad_out_new_grad;
    if (out_grad || grad_x_grad_forward_grad) {
      grad_out_forward_mul_grad_out_new_grad =
          grad_out_forward * grad_out_new_grad.get();
    }

    if (out_grad) {
      auto out_grad_tmp = (scale<T>(grad_x_grad_forward, -2.0) *
                           (grad_out_forward_mul_grad_out_new_grad));
      set_output<T>(out_grad_tmp, out_grad);
    }
    if (grad_out_forward_grad) {
      auto grad_out_forward_grad_tmp =
          (neg_2_out * grad_x_grad_forward * grad_out_new_grad.get());
      set_output<T>(grad_out_forward_grad_tmp, grad_out_forward_grad);
    }
    if (grad_x_grad_forward_grad) {
      auto grad_x_grad_forward_grad_tmp =
          (neg_2_out * grad_out_forward_mul_grad_out_new_grad);
      set_output<T>(grad_x_grad_forward_grad_tmp, grad_x_grad_forward_grad);
    }

  } else if (grad_out_grad_grad) {
    /*
    dy = -2 * y * ddx * dddy
    ddy = 0
    dddx = (1 - y^2) * dddy
    */
    // regard 'grad_out_new_grad' is zero
    if (out_grad) {
      auto out_grad_tmp = (scale<T>(grad_x_grad_forward, -2.0) *
                           (out * grad_out_grad_grad.get()));
      set_output<T>(out_grad_tmp, out_grad);
    }
    if (grad_out_forward_grad) {
      auto grad_out_forward_grad_tmp =
          full<T>(common::vectorize(out.dims()), 0, out.dtype(), out.place());
      set_output<T>(grad_out_forward_grad_tmp, grad_out_forward_grad);
    }
    if (grad_x_grad_forward_grad) {
      auto grad_x_grad_forward_grad_tmp =
          (scale<T>(out * out, -1.0, 1.0) * grad_out_grad_grad.get());
      set_output<T>(grad_x_grad_forward_grad_tmp, grad_x_grad_forward_grad);
    }

  } else {
    /*
    dy = 0
    ddy = 0
    dddx = 0
    */
    if (out_grad) {
      auto out_grad_tmp =
          full<T>(common::vectorize(out.dims()), 0, out.dtype(), out.place());
      set_output<T>(out_grad_tmp, out_grad);
    }
    if (grad_out_forward_grad) {
      auto grad_out_forward_grad_tmp =
          full<T>(common::vectorize(out.dims()), 0, out.dtype(), out.place());
      set_output<T>(grad_out_forward_grad_tmp, grad_out_forward_grad);
    }
    if (grad_x_grad_forward_grad) {
      auto grad_x_grad_forward_grad_tmp =
          full<T>(common::vectorize(grad_x_grad_forward.dims()),
                  0,
                  grad_x_grad_forward.dtype(),
                  grad_x_grad_forward.place());
      set_output<T>(grad_x_grad_forward_grad_tmp, grad_x_grad_forward_grad);
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
  std::vector<std::int64_t> x_dims = common::vectorize(x.dims());
  std::vector<std::int64_t> y_dims = common::vectorize(y.dims());
  std::vector<std::int64_t> grad_out_dims = common::vectorize(grad_out.dims());

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
    return;

  } else if (!grad_x_grad) {
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
  std::vector<int64_t> dx_dims = dx.initialized() ? common::vectorize(dx.dims())
                                                  : std::vector<int64_t>({});
  std::vector<int64_t> dy_dims = dy.initialized() ? common::vectorize(dy.dims())
                                                  : std::vector<int64_t>({});
  std::vector<int64_t> ddout_dims = ddout.initialized()
                                        ? common::vectorize(ddout.dims())
                                        : std::vector<int64_t>({});
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
  auto s = sigmoid<T>(x);
  auto tmp1 = scale<T>(s, -1.0, 1.0);
  auto tmp2 = scale<T>(tmp1 * x, 1.0, 1.0);
  auto grad_x_grad_mul_sigmoid = grad_x_grad * s;
  if (grad_out_grad) {
    auto ddout = grad_x_grad_mul_sigmoid * tmp2;
    set_output<T>(ddout, grad_out_grad);
  }
  if (grad_x) {
    auto dx = grad_x_grad_mul_sigmoid * out_grad *
              (scale<T>(tmp2 - out, 1.0, 1.0)) * tmp1;
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
          auto dx_reduce = dx.sum(common::vectorize(axes),
                                  dx.dtype(),
                                  dx.dims().size() == x.dims().size());
          if (dx_reduce.dims() != x.dims()) {
            dx_reduce = reshape<T>(dx_reduce, x.shape());
          }
          set_output<T>(dx_reduce, x_grad);
        }
      } else {
        set_output<T>(dx, x_grad);
      }

    } else {
      auto dx = full<T>(common::vectorize(x.dims()), 0.0, x.dtype(), x.place());
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
          auto dy_reduce = dy.sum(common::vectorize(axes),
                                  dy.dtype(),
                                  dy.dims().size() == y.dims().size());
          if (dy_reduce.dims() != y.dims()) {
            dy_reduce = reshape<T>(dy_reduce, y.shape());
          }
          set_output<T>(dy_reduce, y_grad);
        }
      } else {
        set_output<T>(dy, y_grad);
      }
    } else {
      auto dy = full<T>(common::vectorize(y.dims()), 0.0, y.dtype(), y.place());
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
      ddout = full<T>(common::vectorize(grad_out.dims()),
                      0.0,
                      grad_out.dtype(),
                      grad_out.place());
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
    if (grad_x_grad && grad_y_grad) {
      set_output<T>(grad_x_grad.get() + grad_y_grad.get(), grad_out_grad);
    } else if (grad_x_grad) {
      by_pass<T>(grad_x_grad.get(), grad_out_grad);
    } else if (grad_y_grad) {
      by_pass<T>(grad_y_grad.get(), grad_out_grad);
    } else {
      set_output<T>(
          full<T>(
              common::vectorize(grad_out.dims()), 0.0, y.dtype(), y.place()),
          grad_out_grad);
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
          auto dddy_reduce_res =
              grad_grad_out_grad.sum(common::vectorize(reduce_dim),
                                     grad_grad_y.get().dtype(),
                                     grad_grad_out_grad.dims().size() ==
                                         grad_grad_y.get().dims().size());
          if (dddy_reduce_res.dims() != grad_grad_y.get().dims()) {
            dddy_reduce_res = reshape<T>(
                dddy_reduce_res, common::vectorize(grad_grad_y.get().dims()));
          }
          set_output<T>(dddy_reduce_res, grad_grad_y_grad);
        }
      } else {
        by_pass<T>(grad_grad_out_grad, grad_grad_y_grad);
      }
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
          auto dddx_reduce_res =
              grad_grad_out_grad.sum(common::vectorize(reduce_dim),
                                     grad_grad_x.get().dtype(),
                                     grad_grad_out_grad.dims().size() ==
                                         grad_grad_x.get().dims().size());
          if (dddx_reduce_res.dims() != grad_grad_x.get().dims()) {
            dddx_reduce_res = reshape<T>(
                dddx_reduce_res, common::vectorize(grad_grad_x.get().dims()));
          }
          set_output<T>(dddx_reduce_res, grad_grad_x_grad);
        }
      } else {
        by_pass<T>(grad_grad_out_grad, grad_grad_x_grad);
      }
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
    if (grad_x_grad && grad_y_grad) {
      set_output<T>(grad_x_grad.get() - grad_y_grad.get(), grad_out_grad);
    } else if (grad_x_grad) {
      if (grad_x_grad.get().dims() != grad_out.dims()) {
        // broad cast grad_x_grad to grad_out
        auto grad_x_grad_dims = common::vectorize(grad_x_grad.get().dims());
        auto grad_out_dims = common::vectorize(grad_out.dims());
        auto broadcast_dims = grad_x_grad_dims;
        // reshape to same dims
        bool need_reshape = false;
        if (grad_out_dims.size() > grad_x_grad_dims.size()) {
          need_reshape = true;
          for (size_t i = 0; i < grad_out_dims.size() - grad_x_grad_dims.size();
               ++i) {
            broadcast_dims.insert(broadcast_dims.begin(), 1);
          }
        }
        // tile if needed
        auto repeat_times = broadcast_dims;
        bool need_tile = false;
        for (size_t i = 0; i < broadcast_dims.size(); ++i) {
          if (grad_out_dims[i] > 1 && broadcast_dims[i] == 1) {
            repeat_times[i] = grad_out_dims[i];
            need_tile = true;
          } else {
            repeat_times[i] = 1;
          }
        }
        if (need_reshape && need_tile) {
          set_output<T>(tile<T>(reshape<T>(grad_x_grad.get(), broadcast_dims),
                                repeat_times),
                        grad_out_grad);
        } else if (need_reshape) {
          set_output<T>(reshape<T>(grad_x_grad.get(), broadcast_dims),
                        grad_out_grad);
        } else if (need_tile) {
          set_output<T>(tile<T>(grad_x_grad.get(), repeat_times),
                        grad_out_grad);
        }
      } else {
        by_pass<T>(grad_x_grad.get(), grad_out_grad);
      }
    } else if (grad_y_grad) {
      if (grad_y_grad.get().dims() != grad_out.dims()) {
        // broad cast grad_y_grad to grad_out
        auto grad_y_grad_dims = common::vectorize(grad_y_grad.get().dims());
        auto grad_out_dims = common::vectorize(grad_out.dims());
        auto broadcast_dims = grad_y_grad_dims;
        // reshape to same dims
        bool need_reshape = false;
        if (grad_out_dims.size() > grad_y_grad_dims.size()) {
          need_reshape = true;
          for (size_t i = 0; i < grad_out_dims.size() - grad_y_grad_dims.size();
               ++i) {
            broadcast_dims.insert(broadcast_dims.begin(), 1);
          }
        }
        // tile if needed
        auto repeat_times = broadcast_dims;
        bool need_tile = false;
        for (size_t i = 0; i < broadcast_dims.size(); ++i) {
          if (grad_out_dims[i] > 1 && broadcast_dims[i] == 1) {
            repeat_times[i] = grad_out_dims[i];
            need_tile = true;
          } else {
            repeat_times[i] = 1;
          }
        }
        if (need_reshape && need_tile) {
          set_output<T>(tile<T>(reshape<T>(grad_y_grad.get(), broadcast_dims),
                                repeat_times),
                        grad_out_grad);
        } else if (need_reshape) {
          set_output<T>(reshape<T>(grad_y_grad.get(), broadcast_dims),
                        grad_out_grad);
        } else if (need_tile) {
          set_output<T>(tile<T>(grad_y_grad.get(), repeat_times),
                        grad_out_grad);
        }
      } else {
        by_pass<T>(-grad_y_grad.get(), grad_out_grad);
      }
    } else {
      set_output<T>(full<T>(common::vectorize(grad_out.dims()),
                            0,
                            grad_out.dtype(),
                            grad_out.place()),
                    grad_out_grad);
    }
  }
}

template <typename T>
void exp_double_grad(const Tensor& out,
                     const Tensor& grad_out,
                     const Tensor& grad_x_grad,
                     Tensor* out_grad,
                     Tensor* grad_out_grad) {
  // dout = dout_old * ddx
  if (out_grad) {
    auto out_grad_tmp = grad_out * grad_x_grad;
    set_output<T>(out_grad_tmp, out_grad);
  }

  // ddout = out * ddx
  if (grad_out_grad) {
    auto grad_out_grad_tmp = out * grad_x_grad;
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }
}

template <typename T>
void log_double_grad(const Tensor& x,
                     const Tensor& grad_out,
                     const Tensor& grad_x_grad,
                     Tensor* x_grad,
                     Tensor* grad_out_grad) {
  // dx = -dout/x^2 * ddx
  if (x_grad) {
    auto x_grad_tmp = -grad_out / (x * x) * grad_x_grad;
    set_output<T>(x_grad_tmp, x_grad);
  }

  // ddout = ddx / x
  if (grad_out_grad) {
    auto grad_out_grad_tmp = grad_x_grad / x;
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }
}

template <typename T>
void abs_triple_grad(const Tensor& x,
                     const Tensor& grad_out_grad_grad,
                     Tensor* grad_grad_x_grad) {
  // dddx = sign(x) * dddout
  if (grad_grad_x_grad) {
    auto grad_grad_x_grad_tmp = sign<T>(x) * grad_out_grad_grad;
    set_output<T>(grad_grad_x_grad_tmp, grad_grad_x_grad);
  }
}

template <typename T>
void bmm_double_grad(const Tensor& x,
                     const Tensor& y,
                     const Tensor& grad_out,
                     const paddle::optional<Tensor>& grad_x_grad,
                     const paddle::optional<Tensor>& grad_y_grad,
                     Tensor* x_grad,
                     Tensor* y_grad,
                     Tensor* grad_out_grad) {
  if (x_grad) {
    // dx' = bmm(dout, ddy.mT)
    Tensor x_grad_tmp;
    if (grad_y_grad) {
      x_grad_tmp =
          matmul<T>(grad_out, transpose<T>(grad_y_grad.get(), {0, 2, 1}));
    } else {
      x_grad_tmp =
          full<T>(common::vectorize(x.dims()), 0, x.dtype(), x.place());
    }
    set_output<T>(x_grad_tmp, x_grad);
  }
  if (y_grad) {
    // dy' = bmm(ddx.mT, dout)
    Tensor y_grad_tmp;
    if (grad_x_grad) {
      y_grad_tmp =
          matmul<T>(transpose<T>(grad_x_grad.get(), {0, 2, 1}), grad_out);
    } else {
      y_grad_tmp =
          full<T>(common::vectorize(y.dims()), 0, y.dtype(), y.place());
    }
    set_output<T>(y_grad_tmp, y_grad);
  }
  if (grad_out_grad) {
    // ddout = bmm(ddx, y) + bmm(x, ddy)
    Tensor grad_out_grad_tmp;
    if (grad_x_grad && grad_y_grad) {
      grad_out_grad_tmp =
          matmul<T>(grad_x_grad.get(), y) + matmul<T>(x, grad_y_grad.get());
    } else if (grad_x_grad) {
      grad_out_grad_tmp = matmul<T>(grad_x_grad.get(), y);
    } else if (grad_y_grad) {
      grad_out_grad_tmp = matmul<T>(x, grad_y_grad.get());
    } else {
      grad_out_grad_tmp = full<T>(common::vectorize(grad_out.dims()),
                                  0,
                                  grad_out.dtype(),
                                  grad_out.place());
    }
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }
}

template <typename T>
void index_put_double_grad(const Tensor& x,
                           const std::vector<Tensor>& indices,
                           const Tensor& value,
                           const paddle::optional<Tensor>& grad_x_grad,
                           const paddle::optional<Tensor>& grad_value_grad,
                           const bool& accumulate,
                           Tensor* grad_out_grad) {
  if (grad_out_grad) {
    if (grad_x_grad && grad_value_grad) {
      /*
        ddout_{i,j} = {
          ddx_{i, j},           (i, j) \notin indices,
          ddv_{k},              (i, j) \in indices and accumulate is false.
          ddx_{i, j} + ddv_{k}, (i, j) \in indices and accumulate is true.
        }
      */
      Tensor grad_out_grad_tmp = grad_x_grad.get();
      grad_out_grad_tmp = index_put<T>(
          grad_out_grad_tmp, indices, grad_value_grad.get(), accumulate);
      set_output<T>(grad_out_grad_tmp, grad_out_grad);

    } else if (grad_x_grad) {
      /*
        ddout_{i,j} = {
          ddx_{i, j},           (i, j) \notin indices,
          0,                    (i, j) \in indices and accumulate is false.
          ddx_{i, j},           (i, j) \in indices and accumulate is true.
        }
      */
      Tensor grad_out_grad_tmp = grad_x_grad.get();
      if (!accumulate) {
        auto zero_to_fill = full<T>(
            common::vectorize(value.dims()), 0, value.dtype(), value.place());
        grad_out_grad_tmp =
            index_put<T>(grad_out_grad_tmp, indices, zero_to_fill, accumulate);
      }
      set_output<T>(grad_out_grad_tmp, grad_out_grad);

    } else if (grad_value_grad) {
      /*
        ddout_{i,j} = {
          0,                    (i, j) \notin indices,
          ddv_{k},              (i, j) \in indices.
        }
      */
      Tensor grad_out_grad_tmp =
          full<T>(common::vectorize(x.dims()), 0, x.dtype(), x.place());
      grad_out_grad_tmp = index_put<T>(grad_out_grad_tmp,
                                       indices,
                                       grad_value_grad.get(),
                                       /*accumulate*/ false);
      set_output<T>(grad_out_grad_tmp, grad_out_grad);

    } else {
      /*
        ddout_{i,j} = 0
      */
      Tensor grad_out_grad_tmp =
          full<T>(common::vectorize(x.dims()), 0, x.dtype(), x.place());
      set_output<T>(grad_out_grad_tmp, grad_out_grad);
    }
  }
}

template <typename T>
void gather_nd_double_grad(const Tensor& grad_out,
                           const Tensor& index,
                           const Tensor& grad_x_grad,
                           Tensor* grad_out_grad) {
  if (grad_out_grad) {
    // ddout = gather_nd(ddx, index)
    auto grad_out_grad_tmp = gather_nd<T>(grad_x_grad, index);
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }
}

template <typename T>
void reshape_double_grad(const Tensor& grad_out,
                         const Tensor& grad_x_grad,
                         Tensor* grad_out_grad) {
  if (grad_out_grad) {
    // ddout = reshape(ddx, ddout.shape)
    auto grad_out_grad_tmp = reshape<T>(grad_x_grad, grad_out.shape());
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }
}

template <typename T>
void take_along_axis_double_grad(const Tensor& indices,
                                 const Tensor& grad_arr_grad,
                                 int axis,
                                 Tensor* grad_out_grad) {
  if (grad_out_grad) {
    if (axis < 0) {
      axis += grad_arr_grad.dims().size();
    }
    // ddout = take_along_axis(ddx, index)
    auto grad_out_grad_tmp = take_along_axis<T>(grad_arr_grad, indices, axis);
    set_output<T>(grad_out_grad_tmp, grad_out_grad);
  }
}

template <typename T>
void index_add_double_grad(const Tensor& index,
                           const Tensor& out_grad,
                           const paddle::optional<Tensor>& grad_x_grad,
                           const paddle::optional<Tensor>& grad_add_value_grad,
                           int axis,
                           Tensor* grad_out_grad) {
  if (grad_out_grad) {
    if (grad_x_grad && grad_add_value_grad) {
      Tensor grad_out_grad_tmp = grad_x_grad.get();
      grad_out_grad_tmp = index_add<T>(
          grad_out_grad_tmp, index, grad_add_value_grad.get(), axis);
      set_output<T>(grad_out_grad_tmp, grad_out_grad);

    } else if (grad_x_grad) {
      Tensor grad_out_grad_tmp = grad_x_grad.get();
      set_output<T>(grad_out_grad_tmp, grad_out_grad);

    } else if (grad_add_value_grad) {
      Tensor DDadd_value = grad_add_value_grad.get();
      Tensor grad_out_grad_tmp = full<T>(common::vectorize(out_grad.dims()),
                                         0,
                                         DDadd_value.dtype(),
                                         DDadd_value.place());
      grad_out_grad_tmp =
          index_add<T>(grad_out_grad_tmp, index, DDadd_value, axis);
      set_output<T>(grad_out_grad_tmp, grad_out_grad);

    } else {
      Tensor grad_out_grad_tmp = full<T>(common::vectorize(out_grad.dims()),
                                         0,
                                         out_grad.dtype(),
                                         out_grad.place());
      set_output<T>(grad_out_grad_tmp, grad_out_grad);
    }
  }
}

}  // namespace prim
}  // namespace paddle
