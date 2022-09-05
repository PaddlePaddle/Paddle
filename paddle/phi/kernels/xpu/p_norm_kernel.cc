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

#include "paddle/phi/kernels/p_norm_kernel.h"
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
void PNormKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 float porder,
                 int axis,
                 float epsilon,
                 bool keepdim,
                 bool asvector,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);
  auto xdim = x.dims();
  if (axis < 0) axis = xdim.size() + axis;
  std::vector<int> r_dim;
  std::vector<int> x_dim;
  std::vector<int> y_dim;
  int m = 1;
  int n = 1;
  int t = 1;
  GetDims(xdim, axis, &m, &t, &n, asvector);
  x_dim.push_back(m);
  x_dim.push_back(t);
  x_dim.push_back(n);

  r_dim.push_back(1);

  y_dim.push_back(m);
  y_dim.push_back(n);

  int r = 0;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  XPUType* tmp_x = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
  PADDLE_ENFORCE_XDNN_NOT_NULL(tmp_x);

  r = xpu::abs(dev_ctx.x_context(),
               reinterpret_cast<const XPUType*>(x.data<T>()),
               tmp_x,
               m * t * n);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "abs");
  if (porder == INFINITY) {
    r = xpu::reduce_max(dev_ctx.x_context(),
                        tmp_x,
                        reinterpret_cast<XPUType*>(out->data<T>()),
                        x_dim,
                        r_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_max");
  } else if (porder == -INFINITY) {
    r = xpu::reduce_min(dev_ctx.x_context(),
                        tmp_x,
                        reinterpret_cast<XPUType*>(out->data<T>()),
                        x_dim,
                        r_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_min");
  } else if (porder == 0) {
    XPUType* zeros = RAII_GUARD.alloc_l3_or_gm<XPUType>(1);
    PADDLE_ENFORCE_XDNN_NOT_NULL(zeros);
    r = xpu::constant(dev_ctx.x_context(), zeros, 1, 0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    std::vector<int> zeros_dim(1, 1);

    bool* tmp2_x = RAII_GUARD.alloc_l3_or_gm<bool>(m * t * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(tmp2_x);

    r = xpu::broadcast_not_equal(
        dev_ctx.x_context(), tmp_x, zeros, tmp2_x, x_dim, zeros_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_not_equal");

    XPUType* x_mid = tmp_x;

    r = xpu::cast<bool, T>(dev_ctx.x_context(), tmp2_x, x_mid, m * t * n);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");

    r = xpu::reduce_sum(dev_ctx.x_context(),
                        x_mid,
                        reinterpret_cast<XPUType*>(out->data<T>()),
                        x_dim,
                        r_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

  } else {
    DenseTensor porder_tensor;
    phi::DDim pdim = phi::make_ddim({1});
    porder_tensor.Resize(pdim);
    dev_ctx.template Alloc<T>(&porder_tensor);
    r = xpu::constant(
        dev_ctx.x_context(), porder_tensor.data<float>(), 1, porder);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    std::vector<int> p_dim(1, 1);

    XPUType* tmp2_x = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(tmp2_x);
    r = xpu::broadcast_pow(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(tmp_x),
        reinterpret_cast<const XPUType*>(porder_tensor.data<float>()),
        reinterpret_cast<XPUType*>(tmp2_x),
        x_dim,
        p_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_pow");

    XPUType* tmp_y = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * n);
    PADDLE_ENFORCE_XDNN_NOT_NULL(tmp_y);

    r = xpu::reduce_sum(dev_ctx.x_context(),
                        reinterpret_cast<const XPUType*>(tmp2_x),
                        reinterpret_cast<XPUType*>(tmp_y),
                        x_dim,
                        r_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

    r = xpu::constant(
        dev_ctx.x_context(), porder_tensor.data<float>(), 1, 1.0f / porder);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

    r = xpu::broadcast_pow(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(tmp_y),
        reinterpret_cast<const XPUType*>(porder_tensor.data<float>()),
        reinterpret_cast<XPUType*>(out->data<T>()),
        y_dim,
        p_dim);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_pow");
    dev_ctx.Wait();
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(p_norm, XPU, ALL_LAYOUT, phi::PNormKernel, float) {}
