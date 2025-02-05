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

#include "paddle/phi/kernels/p_norm_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

inline void GetDims(
    const phi::DDim& dim, int axis, int* m, int* t, int* n, bool asvector) {
  *m = 1;
  *n = 1;
  *t = dim[axis];
  if (asvector) {
    *t = product(dim);
  } else {
    for (int i = 0; i < axis; ++i) {
      (*m) *= dim[i];
    }
    for (int i = axis + 1; i < dim.size(); ++i) {
      (*n) *= dim[i];
    }
  }
}
template <typename T, typename Context>
void PNormGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out,
                     const DenseTensor& out_grad,
                     float porder,
                     int axis,
                     float epsilon,
                     bool keepdim,
                     bool asvector,
                     DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(x_grad);
  auto xdim = x.dims();
  axis = axis < 0 ? xdim.size() + axis : axis;
  int m, t, n;
  GetDims(xdim, axis, &m, &t, &n, asvector);

  std::vector<int> r_dim;
  std::vector<int> x_dim;
  std::vector<int> y_dim;

  x_dim.push_back(m);
  x_dim.push_back(t);
  x_dim.push_back(n);

  y_dim.push_back(m);
  y_dim.push_back(1);
  y_dim.push_back(n);

  int r = 0;
  if (porder == 0) {
    r = xpu::constant(dev_ctx.x_context(),
                      reinterpret_cast<XPUType*>(x_grad->data<T>()),
                      m * t * n,
                      static_cast<T>(0));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  } else if (porder == INFINITY || porder == -INFINITY) {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    XPUType* x_abs = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(x_abs);
    r = xpu::abs(dev_ctx.x_context(),
                 reinterpret_cast<const XPUType*>(x.data<T>()),
                 x_abs,
                 m * t * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "abs");

    bool* dx_t = RAII_GUARD.alloc_l3_or_gm<bool>(m * t * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(dx_t);

    XPUType* dx_mid = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(dx_mid);

    r = xpu::broadcast_equal<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x_abs),
        reinterpret_cast<const XPUType*>(out.data<T>()),
        dx_t,
        x_dim,
        y_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_equal");

    r = xpu::cast<bool, XPUType>(dev_ctx.x_context(), dx_t, dx_mid, m * t * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");

    XPUType* x_sign = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(x_sign);
    r = xpu::sign(dev_ctx.x_context(),
                  reinterpret_cast<const XPUType*>(x.data<T>()),
                  x_sign,
                  m * t * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sign");

    XPUType* dx_pre_dy = x_abs;
    r = xpu::mul(dev_ctx.x_context(),
                 reinterpret_cast<const XPUType*>(dx_mid),
                 reinterpret_cast<const XPUType*>(x_sign),
                 dx_pre_dy,
                 m * t * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");

    r = xpu::broadcast_mul(dev_ctx.x_context(),
                           dx_pre_dy,
                           reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                           reinterpret_cast<XPUType*>(x_grad->data<T>()),
                           x_dim,
                           y_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

  } else {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    XPUType* x_abs = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(x_abs);
    r = xpu::abs(dev_ctx.x_context(),
                 reinterpret_cast<const XPUType*>(x.data<T>()),
                 x_abs,
                 m * t * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "abs");

    DenseTensor porder_tensor;
    phi::DDim pdim = common::make_ddim({1});
    porder_tensor.Resize(pdim);
    dev_ctx.template Alloc<float>(&porder_tensor);
    r = xpu::constant(
        dev_ctx.x_context(), porder_tensor.data<float>(), 1, porder - 1.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    std::vector<int> p_dim(1, 1);

    XPUType* x_pow = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(x_pow);
    r = xpu::broadcast_pow(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x_abs),
        reinterpret_cast<const XPUType*>(porder_tensor.data<float>()),
        x_pow,
        x_dim,
        p_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_pow");

    XPUType* y_pow = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(y_pow);
    r = xpu::broadcast_pow(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(out.data<T>()),
        reinterpret_cast<const XPUType*>(porder_tensor.data<float>()),
        y_pow,
        y_dim,
        p_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_pow");
    dev_ctx.Wait();

    XPUType* dx_t = x_abs;

    r = xpu::broadcast_div(
        dev_ctx.x_context(), x_pow, y_pow, dx_t, x_dim, y_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_div");

    XPUType* x_sign = x_pow;
    r = xpu::sign(dev_ctx.x_context(),
                  reinterpret_cast<const XPUType*>(x.data<T>()),
                  x_sign,
                  m * t * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sign");

    XPUType* dx_mid = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(dx_mid);

    r = xpu::broadcast_mul(dev_ctx.x_context(),
                           reinterpret_cast<const XPUType*>(x_sign),
                           reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                           dx_mid,
                           x_dim,
                           y_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

    r = xpu::broadcast_mul(dev_ctx.x_context(),
                           reinterpret_cast<const XPUType*>(dx_t),
                           reinterpret_cast<const XPUType*>(dx_mid),
                           reinterpret_cast<XPUType*>(x_grad->data<T>()),
                           x_dim,
                           x_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(p_norm_grad, XPU, ALL_LAYOUT, phi::PNormGradKernel, float) {}
