//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU

#include <string>
#include <vector>
#include "paddle/fluid/operators/matmul_v2_op.h"

#include "paddle/fluid/operators/xpu_api_wrapper.h"

namespace paddle {
namespace operators {

template <typename T>
class MatMulV2XPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");
    out->mutable_data<T>(ctx.GetPlace());
    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y->data<T>());
    XPUType* out_ptr = reinterpret_cast<XPUType*>(out->data<T>());
    auto x_dims = x->dims();
    auto y_dims = y->dims();

    XpuFcInfo fc_info;
    GetFCInfo(x_dims, y_dims, trans_x, trans_y, &fc_info);
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    xpu::Context* xpu_ctx = dev_ctx.x_context();
    MatMulXPUFunction<XPUType>(xpu_ctx, x_ptr, y_ptr, out_ptr, fc_info, 1.0f);
  }
};

template <typename T>
class MatMulV2XPUGradKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool transpose_x = context.Attr<bool>("trans_x");
    bool transpose_y = context.Attr<bool>("trans_y");
    auto x = *context.Input<framework::Tensor>("X");
    auto y = *context.Input<framework::Tensor>("Y");
    auto dout =
        *context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = context.Output<framework::Tensor>(framework::GradVarName("Y"));
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

    XpuFcInfo info_forward;
    GetFCInfo(x.dims(), y.dims(), transpose_x, transpose_y, &info_forward);
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
                                   transpose_x,
                                   transpose_y,
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
REGISTER_OP_XPU_KERNEL(matmul_v2,
                       ops::MatMulV2XPUKernel<float>,
                       ops::MatMulV2XPUKernel<plat::float16>);
REGISTER_OP_XPU_KERNEL(matmul_v2_grad,
                       ops::MatMulV2XPUGradKernel<float>,
                       ops::MatMulV2XPUGradKernel<plat::float16>);

#endif
