/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

template <typename T>
struct XPUAddFunctor {
  int operator()(xpu::Context* ctx, const T* x, const T* y, T* z, int len) {
    return xpu::elementwise_add(ctx, x, y, z, len);
  }
};

template <typename T>
struct XPUMulFunctor {
  int operator()(xpu::Context* ctx, const T* x, const T* y, T* z, int len) {
    return xpu::elementwise_mul(ctx, x, y, z, len);
  }
};

template <typename T, typename Functor>
void XPUElementwise(const framework::ExecutionContext& ctx) {
  PADDLE_ENFORCE(platform::is_xpu_place(ctx.GetPlace()),
                 "This kernel only runs on XPU device.");
  auto x_var = ctx.InputVar("X");
  PADDLE_ENFORCE_NE(x_var, nullptr,
                    platform::errors::Fatal("Cannot get input Variable X"));
  PADDLE_ENFORCE(x_var->IsType<framework::LoDTensor>(),
                 "XPU only support LoDTensor");

  auto x = x_var->Get<framework::LoDTensor>();
  auto* y = ctx.Input<framework::LoDTensor>("Y");
  auto* z = ctx.Output<framework::LoDTensor>("Out");
  z->mutable_data<T>(ctx.GetPlace());

  int axis = ctx.Attr<int>("axis");
  auto x_dims = x.dims();
  auto y_dims_untrimed = y->dims();
  PADDLE_ENFORCE_GE(x_dims.size(), y_dims_untrimed.size(),
                    "Rank of first input must >= rank of second input.");
  axis = (axis == -1 ? x_dims.size() - y_dims_untrimed.size() : axis);
  PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                 "Axis should be in range [0, x_dims)");
  auto y_dims = trim_trailing_singular_dims(y_dims_untrimed);
  axis = (y_dims.size() == 0) ? x_dims.size() : axis;
  int pre, n, post, is_common_broadcast;
  get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post, &is_common_broadcast);
  int len = pre * n * post;

  const T* x_data = x.data<T>();
  const T* y_data = y->data<T>();
  T* z_data = z->data<T>();
  T* y_broadcast = nullptr;

  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();

  if (post == 1) {
    if (std::is_same<Functor, XPUAddFunctor<T>>::value) {
      int res = xpu::matrix_vector_add(dev_ctx.x_context(), x_data, y_data,
                                       z_data, pre, n);
      PADDLE_ENFORCE(res == xpu::Error_t::SUCCESS, "XPU kernel error! res = %d",
                     res);
      return;
    }
    if (std::is_same<Functor, XPUMulFunctor<T>>::value) {
      int res = xpu::matrix_vector_mul(dev_ctx.x_context(), x_data, y_data,
                                       z_data, pre, n);
      PADDLE_ENFORCE(res == xpu::Error_t::SUCCESS, "XPU kernel error! res = %d",
                     res);
      return;
    }
  }

  if (pre != 1 || post != 1) {
    PADDLE_ENFORCE(xpu_malloc(reinterpret_cast<void**>(&y_broadcast),
                              len * sizeof(T)) == XPU_SUCCESS);
    int res = xpu::broadcast_ew(dev_ctx.x_context(), y_data, y_broadcast, pre,
                                n, post, xpu::ElementwiseOp::ASSIGN);
    PADDLE_ENFORCE(res == xpu::Error_t::SUCCESS, "XPU kernel error! res = %d",
                   res);
    y_data = y_broadcast;
  }

  Functor functor;
  int res = functor(dev_ctx.x_context(), x_data, y_data, z_data, len);
  PADDLE_ENFORCE(res == xpu::Error_t::SUCCESS, "XPU kernel error! res = %d",
                 res);

  if (pre != 1 || post != 1) {
    dev_ctx.Wait();
    xpu_free(y_broadcast);
  }
}

}  // namespace operators
}  // namespace paddle
#endif
