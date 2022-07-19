/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op_xpu.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

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

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class P_NormXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    float porder = ctx.Attr<float>("porder");
    int axis = ctx.Attr<int>("axis");
    bool asvector = ctx.Attr<bool>("asvector");

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto xdim = in->dims();
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
                 reinterpret_cast<const XPUType*>(in->data<T>()),
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
      std::vector<int> zeros_dim(1, 1);

      bool* tmp2_x = RAII_GUARD.alloc_l3_or_gm<bool>(m * t * n);
      PADDLE_ENFORCE_XDNN_NOT_NULL(tmp2_x);

      r = xpu::broadcast_not_equal(
          dev_ctx.x_context(), tmp_x, zeros, tmp2_x, x_dim, zeros_dim);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_not_equal");

      XPUType* x_mid = tmp_x;

      r = xpu::cast<bool, XPUType>(
          dev_ctx.x_context(), tmp2_x, x_mid, m * t * n);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");

      r = xpu::reduce_sum(dev_ctx.x_context(),
                          x_mid,
                          reinterpret_cast<XPUType*>(out->data<T>()),
                          x_dim,
                          r_dim);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

    } else {
      Tensor porder_tensor;
      framework::DDim pdim = phi::make_ddim({1});
      porder_tensor.mutable_data<float>(pdim, in->place());
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
          tmp2_x,
          x_dim,
          p_dim);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_pow");

      XPUType* tmp_y = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * n);
      PADDLE_ENFORCE_XDNN_NOT_NULL(tmp_y);

      r = xpu::reduce_sum(dev_ctx.x_context(),
                          reinterpret_cast<const XPUType*>(tmp2_x),
                          tmp_y,
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
};

template <typename DeviceContext, typename T>
class P_NormGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Out");
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto xdim = x->dims();
    float porder = ctx.Attr<float>("porder");
    bool asvector = ctx.Attr<bool>("asvector");
    int axis = ctx.Attr<int>("axis");
    axis = axis < 0 ? xdim.size() + axis : axis;

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

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
                        reinterpret_cast<XPUType*>(dx->data<T>()),
                        m * t * n,
                        static_cast<T>(0));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    } else if (porder == INFINITY || porder == -INFINITY) {
      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      XPUType* x_abs = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
      PADDLE_ENFORCE_XDNN_NOT_NULL(x_abs);
      r = xpu::abs(dev_ctx.x_context(),
                   reinterpret_cast<const XPUType*>(x->data<T>()),
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
          reinterpret_cast<const XPUType*>(y->data<T>()),
          dx_t,
          x_dim,
          y_dim);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_equal");

      r = xpu::cast<bool, XPUType>(
          dev_ctx.x_context(), dx_t, dx_mid, m * t * n);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");

      XPUType* x_sign = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
      PADDLE_ENFORCE_XDNN_NOT_NULL(x_sign);
      r = xpu::sign(dev_ctx.x_context(),
                    reinterpret_cast<const XPUType*>(x->data<T>()),
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
                             reinterpret_cast<const XPUType*>(dy->data<T>()),
                             reinterpret_cast<XPUType*>(dx->data<T>()),
                             x_dim,
                             y_dim);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

    } else {
      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      XPUType* x_abs = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
      PADDLE_ENFORCE_XDNN_NOT_NULL(x_abs);
      r = xpu::abs(dev_ctx.x_context(),
                   reinterpret_cast<const XPUType*>(x->data<T>()),
                   x_abs,
                   m * t * n);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "abs");

      Tensor porder_tensor;
      framework::DDim pdim = phi::make_ddim({1});
      porder_tensor.mutable_data<float>(pdim, x->place());
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
          reinterpret_cast<const XPUType*>(y->data<T>()),
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
                    reinterpret_cast<const XPUType*>(x->data<T>()),
                    x_sign,
                    m * t * n);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "sign");

      XPUType* dx_mid = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * t * n);
      PADDLE_ENFORCE_XDNN_NOT_NULL(dx_mid);

      r = xpu::broadcast_mul(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x_sign),
                             reinterpret_cast<const XPUType*>(dy->data<T>()),
                             dx_mid,
                             x_dim,
                             y_dim);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

      r = xpu::broadcast_mul(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(dx_t),
                             reinterpret_cast<const XPUType*>(dx_mid),
                             reinterpret_cast<XPUType*>(dx->data<T>()),
                             x_dim,
                             x_dim);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    p_norm, ops::P_NormXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    p_norm_grad,
    ops::P_NormGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
