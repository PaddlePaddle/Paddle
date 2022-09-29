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

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/xpu_api_wrapper.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MatMulXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<phi::DenseTensor>("X");
    auto* y = context.Input<phi::DenseTensor>("Y");
    auto* out = context.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(context.GetPlace());
    bool trans_x = context.Attr<bool>("transpose_X");
    bool trans_y = context.Attr<bool>("transpose_Y");
    float alpha = static_cast<T>(context.Attr<float>("alpha"));
    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y->data<T>());
    XPUType* out_ptr = reinterpret_cast<XPUType*>(out->data<T>());
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    phi::XpuFcInfo fc_info;
    phi::GetFCInfo(x_dims, y_dims, trans_x, trans_y, &fc_info);
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    xpu::Context* xpu_ctx = dev_ctx.x_context();

    phi::MatMulXPUFunction<XPUType>(
        xpu_ctx, x_ptr, y_ptr, out_ptr, fc_info, alpha);
  }
};

// Using dimensional constraints on matrix multiplication, it is
// straight-forward to check the following table for when X and Y
// are both matrices.
//
// transpose_X | False    | True     | False    | True
// transpose_Y | False    | False    | True     | True
// -----------+----------+----------+----------+-----------
//        dX = | dOut Y^T | Y dOut^T | dOut Y   | Y^T dOut^T
//        dY = | X^T dOut | X dOut   | dOut^T X | dOut^T X^T
//
// When X is a vector of size K, we treat it instead as a matrix of shape
// (1, K). Similarly, when Y is a vector of size K, we treat it instead as
// a matrix of shape (K, 1).
//
// When X and Y are both 3-dimensional tensors, then the first dimension
// the batch dimension can be ignored and the exact same formulas apply
// as for two matrices.
//
// Finally, when, e.g., X is a 3-dimensional tensor but Y is a matrix, we end
// up with formulas like
//
//   dY_{ij} = \sum_{p, m} X_{pmi} dOut_{pmj}
//
// To handle this sort of scenario, we reshape X : P x M x K, dOut: P x M x N
// to X: (P * M) x K, dOut: (P * M) x N.
template <typename DeviceContext, typename T>
class MatMulGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto x = *context.Input<phi::DenseTensor>("X");
    auto y = *context.Input<phi::DenseTensor>("Y");
    auto dout = *context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = context.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    bool transpose_x = context.Attr<bool>("transpose_X");
    bool transpose_y = context.Attr<bool>("transpose_Y");
    float alpha = static_cast<T>(context.Attr<float>("alpha"));
    if (dx) {
      dx->mutable_data<T>(context.GetPlace());
    }
    if (dy) {
      dy->mutable_data<T>(context.GetPlace());
    }
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();

    const XPUType* dout_ptr = reinterpret_cast<const XPUType*>(dout.data<T>());
    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
    const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());

    xpu::Context* xpu_ctx = dev_ctx.x_context();

    phi::XpuFcInfo info_forward;
    phi::GetFCInfo(x.dims(), y.dims(), transpose_x, transpose_y, &info_forward);
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    // begin calculate
    const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
    const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
    const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
    const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
    XPUType* c_1 = (dx == NULL) ? reinterpret_cast<XPUType*>(NULL)
                                : reinterpret_cast<XPUType*>(dx->data<T>());
    XPUType* c_2 = (dy == NULL) ? reinterpret_cast<XPUType*>(NULL)
                                : reinterpret_cast<XPUType*>(dy->data<T>());
    phi::XpuFcInfo info_dx;
    phi::XpuFcInfo info_dy;
    std::tuple<phi::XpuFcInfo,
               phi::XpuFcInfo,
               const XPUType*,
               const XPUType*,
               const XPUType*,
               const XPUType*>
        fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                        &RAII_GUARD,
                                        info_forward,
                                        transpose_x,
                                        transpose_y,
                                        x_ptr,
                                        y_ptr,
                                        dout_ptr);
    std::tie(info_dx, info_dy, a_1, b_1, a_2, b_2) = fc_info;
    if (dx) {
      phi::MatMulXPUFunction<XPUType>(xpu_ctx, a_1, b_1, c_1, info_dx, alpha);
    }
    if (dy) {
      phi::MatMulXPUFunction<XPUType>(xpu_ctx, a_2, b_2, c_2, info_dy, alpha);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    matmul,
    ops::MatMulXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MatMulXPUKernel<paddle::platform::XPUDeviceContext, plat::float16>);
REGISTER_OP_XPU_KERNEL(
    matmul_grad,
    ops::MatMulGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MatMulGradXPUKernel<paddle::platform::XPUDeviceContext,
                             plat::float16>);
#endif
