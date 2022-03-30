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

#ifdef PADDLE_WITH_XPU

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/xpu_api_wrapper.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

template <typename DeviceContext, typename T>
class MulXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");
    Tensor* z = context.Output<Tensor>("Out");
    const Tensor x_matrix =
        x->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *x, context.template Attr<int>("x_num_col_dims"))
            : *x;
    const Tensor y_matrix =
        y->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *y, context.template Attr<int>("y_num_col_dims"))
            : *y;
    z->mutable_data<T>(context.GetPlace());
    auto z_dim = z->dims();
    if (z_dim.size() != 2) {
      z->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }
    bool trans_a = false;
    bool trans_b = false;
    int m = x_matrix.dims()[0];
    int k = x_matrix.dims()[1];
    int k1 = y_matrix.dims()[0];
    int n = y_matrix.dims()[1];
    PADDLE_ENFORCE_EQ(
        k, k1, platform::errors::InvalidArgument("Shape mistake in mul_op"));
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    const T* data_a = x_matrix.data<T>();
    const T* data_b = y_matrix.data<T>();
    T* data_c = z->data<T>();
    auto& dev_ctx = context.template device_context<DeviceContext>();

    int ret = xpu_fc_wrapper<XPUType, int16_t>(
        dev_ctx.x_context(), reinterpret_cast<const XPUType*>(data_a),
        reinterpret_cast<const XPUType*>(data_b),
        reinterpret_cast<XPUType*>(data_c), m, n, k, trans_a, trans_b, nullptr,
        nullptr, nullptr, k, n, n, alpha, beta, nullptr,
        xpu::Activation_t::LINEAR);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xpu_fc_wrapper");

    if (z_dim.size() != 2) {
      z->Resize(z_dim);
    }
  }
};

template <typename DeviceContext, typename T>
class MulGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int x_num_col_dims = ctx.template Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.template Attr<int>("y_num_col_dims");
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto x_matrix = x->dims().size() > 2
                        ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                        : static_cast<const Tensor&>(*x);
    auto y_matrix = y->dims().size() > 2
                        ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                        : static_cast<const Tensor&>(*y);
    auto* dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    Tensor dout_mat;
    dout_mat.Resize({phi::flatten_to_2d(x->dims(), x_num_col_dims)[0],
                     phi::flatten_to_2d(y->dims(), y_num_col_dims)[1]});
    auto* dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));
    if (dx != nullptr) {
      dx->set_lod(x->lod());
    }
    if (dy != nullptr) {
      dy->set_lod(y->lod());
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      Tensor dx_matrix = dx->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dx, x_num_col_dims)
                             : *dx;
      // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
      // blas.MatMul(dout_mat, false, y_matrix, true, &dx_matrix);
      bool trans_a = false;
      bool trans_b = true;
      int m = dout_mat.dims()[0];
      int k = dout_mat.dims()[1];
      int n = y_matrix.dims()[0];
      int k1 = y_matrix.dims()[1];
      PADDLE_ENFORCE_EQ(
          k, k1, platform::errors::InvalidArgument("Shape mistake in mul_op"));
      int lda = (!trans_a) ? k : m;
      int ldb = (!trans_b) ? n : k;
      int ldc = n;
      T alpha = static_cast<T>(1.0);
      T beta = static_cast<T>(0.0);
      const T* data_a = dout->data<T>();
      const T* data_b = y_matrix.data<T>();
      T* data_c = dx_matrix.data<T>();

      int ret = xpu_fc_wrapper<XPUType, int16_t>(
          dev_ctx.x_context(), reinterpret_cast<const XPUType*>(data_a),
          reinterpret_cast<const XPUType*>(data_b),
          reinterpret_cast<XPUType*>(data_c), m, n, k, trans_a, trans_b,
          nullptr, nullptr, nullptr, lda, ldb, ldc, alpha, beta, nullptr,
          xpu::Activation_t::LINEAR);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xpu_fc_wrapper");
    }

    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor dy_matrix = dy->dims().size() > 2
                             ? framework::ReshapeToMatrix(*dy, y_num_col_dims)
                             : *dy;
      // dy = x' * dout. dy K x N, dout : M x N, x : M x K
      // blas.MatMul(x_matrix, true, dout_mat, false, &dy_matrix);
      bool trans_a = true;
      bool trans_b = false;
      int k = x_matrix.dims()[0];
      int m = x_matrix.dims()[1];
      int k1 = dout_mat.dims()[0];
      int n = dout_mat.dims()[1];
      PADDLE_ENFORCE_EQ(
          k, k1, platform::errors::InvalidArgument("Shape mistake in mul_op"));
      int lda = (!trans_a) ? k : m;
      int ldb = (!trans_b) ? n : k;
      int ldc = n;
      T alpha = static_cast<T>(1.0);
      T beta = static_cast<T>(0.0);
      const T* data_a = x_matrix.data<T>();
      const T* data_b = dout->data<T>();
      T* data_c = dy_matrix.data<T>();

      int ret = xpu_fc_wrapper<XPUType, int16_t>(
          dev_ctx.x_context(), reinterpret_cast<const XPUType*>(data_a),
          reinterpret_cast<const XPUType*>(data_b),
          reinterpret_cast<XPUType*>(data_c), m, n, k, trans_a, trans_b,
          nullptr, nullptr, nullptr, lda, ldb, ldc, alpha, beta, nullptr,
          xpu::Activation_t::LINEAR);
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xpu_fc_wrapper");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    mul, ops::MulXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MulXPUKernel<paddle::platform::XPUDeviceContext, plat::float16>);
REGISTER_OP_XPU_KERNEL(
    mul_grad, ops::MulGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MulGradXPUKernel<paddle::platform::XPUDeviceContext, plat::float16>)
#endif
