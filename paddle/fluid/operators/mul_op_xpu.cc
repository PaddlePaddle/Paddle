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

    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x_matrix.data<T>());
    const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y_matrix.data<T>());
    XPUType* out_ptr = reinterpret_cast<XPUType*>(z->data<T>());

    bool trans_a = false;
    bool trans_b = false;
    auto x_dims = x_matrix.dims();
    auto y_dims = y_matrix.dims();

    XpuFcInfo fc_info;
    GetFCInfo(x_dims, y_dims, trans_a, trans_b, &fc_info);
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    xpu::Context* xpu_ctx = dev_ctx.x_context();

    MatMulXPUFunction<XPUType>(xpu_ctx, x_ptr, y_ptr, out_ptr, fc_info, 1.0f);
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

    XpuFcInfo info_forward;
    GetFCInfo(x_matrix.dims(), y_matrix.dims(), false, false, &info_forward);

    const XPUType* dout_ptr = reinterpret_cast<const XPUType*>(dout->data<T>());
    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y->data<T>());

    xpu::Context* xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    // begin calculate
    const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
    const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
    const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
    const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
    XPUType* c_1 =
        (dx == NULL)
            ? reinterpret_cast<XPUType*>(NULL)
            : reinterpret_cast<XPUType*>(dx->mutable_data<T>(ctx.GetPlace()));
    XPUType* c_2 =
        (dy == NULL)
            ? reinterpret_cast<XPUType*>(NULL)
            : reinterpret_cast<XPUType*>(dy->mutable_data<T>(ctx.GetPlace()));
    XpuFcInfo info_dx;
    XpuFcInfo info_dy;
    std::tuple<XpuFcInfo,
               XpuFcInfo,
               const XPUType*,
               const XPUType*,
               const XPUType*,
               const XPUType*>
        fc_info = MatmulGradFcInfo(xpu_ctx,
                                   &RAII_GUARD,
                                   info_forward,
                                   false,
                                   false,
                                   x_ptr,
                                   y_ptr,
                                   dout_ptr);
    std::tie(info_dx, info_dy, a_1, b_1, a_2, b_2) = fc_info;
    if (dx) {
      MatMulXPUFunction<XPUType>(xpu_ctx, a_1, b_1, c_1, info_dx, 1.0f);
    }
    if (dy) {
      MatMulXPUFunction<XPUType>(xpu_ctx, a_2, b_2, c_2, info_dy, 1.0f);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    mul,
    ops::MulXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MulXPUKernel<paddle::platform::XPUDeviceContext, plat::float16>);
REGISTER_OP_XPU_KERNEL(
    mul_grad,
    ops::MulGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MulGradXPUKernel<paddle::platform::XPUDeviceContext, plat::float16>)
#endif
