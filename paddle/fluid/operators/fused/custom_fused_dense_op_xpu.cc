/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class CustomFusedDenseXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  // using biasType= typename XPUTypeTrait<float>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();
    auto* xpu_ctx = dev_ctx.x_context();

    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor* bias = ctx.Input<phi::DenseTensor>("Bias");

    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    phi::DenseTensor* gelu_in =
        ctx.Output<phi::DenseTensor>("GeluIn");

    bool transx = ctx.Attr<bool>("transx");
    bool transy = ctx.Attr<bool>("transy");

    std::string activation = ctx.Attr<std::string>("activation");
    VLOG(5) << "transx = " << transx << " , transy = " << transy
            << " , activation = " << activation;

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    int m = transx ? x_dims[1] : x_dims[0]; 
    int k = transx ? x_dims[0] : x_dims[1];
    int n = transy ? y_dims[0] : y_dims[1];

    // 调用新接口，这里先分开调用，等待qingpen的新接口
    int r = 0;
    xpu::Activation_t act = xpu::Activation_t::LINEAR;
    if (activation == "relu") {
      act = xpu::Activation_t::RELU;
    } else if (activation == "gelu") {
      act = xpu::Activation_t::GELU;
    }

    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y->data<T>());
    XPUType* out_ptr =
        reinterpret_cast<XPUType*>(out->mutable_data<T>(ctx.GetPlace()));

    XPUType* gelu_in_ptr = nullptr;
    if (activation != "none" && gelu_in) {
      gelu_in_ptr = reinterpret_cast<XPUType*>(
          gelu_in->mutable_data<T>(ctx.GetPlace()));
    }

    const XPUType* bias_ptr = reinterpret_cast<const XPUType*>(bias->data<T>());
    
    r = xpu::fused_dense_fusion<XPUType, XPUType, XPUType, XPUType, XPUType, int16_t>(
            xpu_ctx, x_ptr, y_ptr, out_ptr, m, n, k,
            transx, transy, nullptr, nullptr, nullptr,
            x_dims[1], y_dims[1], n, 1.0f, 0.0f, bias_ptr, act, gelu_in_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "fused_dense_fusion");
  }
};

template <typename DeviceContext, typename T>
class CustomFusedDenseXPUGradKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int r = -1;
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();
    auto* xpu_ctx = dev_ctx.x_context();

    bool transx = ctx.Attr<bool>("transx");
    bool transy = ctx.Attr<bool>("transy");
    // bool use_addto = ctx.Attr<bool>("use_addto");

    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor* dout = ctx.Input<phi::DenseTensor>("DOut");

    const phi::DenseTensor* gelu_in =
        ctx.Input<phi::DenseTensor>("GeluIn");

    phi::DenseTensor* dx = ctx.Output<phi::DenseTensor>("DX");
    phi::DenseTensor* dy = ctx.Output<phi::DenseTensor>("DY");
    phi::DenseTensor* dbias = ctx.Output<phi::DenseTensor>("DBias");

    std::string activation = "none";
    if (ctx.HasAttr("activation")) {
      activation = ctx.Attr<std::string>("activation");
    } else if (ctx.HasAttr("activation_grad")) {
      activation = ctx.Attr<std::string>("activation_grad");
    }
    xpu::Activation_t act = xpu::Activation_t::LINEAR;
    if (activation == "relu") {
      act = xpu::Activation_t::RELU;
    } else if (activation == "gelu") {
      act = xpu::Activation_t::GELU;
    }

    const XPUType* dout_ptr = reinterpret_cast<const XPUType*>(dout->data<T>());

    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y->data<T>());

    // const XPUType*
    const XPUType* gelu_in_ptr =
        (gelu_in == NULL)
            ? (reinterpret_cast<const XPUType*>(NULL))
            : (reinterpret_cast<const XPUType*>(gelu_in->data<T>()));

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    int m = transx ? x_dims[1] : x_dims[0]; 
    int k = transx ? x_dims[0] : x_dims[1];
    int n = transy ? y_dims[0] : y_dims[1];

    // 2. fc_grad
    XPUType* dx_ptr =
        (dx == NULL)
            ? reinterpret_cast<XPUType*>(NULL)
            : reinterpret_cast<XPUType*>(dx->mutable_data<T>(ctx.GetPlace()));
    XPUType* dy_ptr =
        (dy == NULL)
            ? reinterpret_cast<XPUType*>(NULL)
            : reinterpret_cast<XPUType*>(dy->mutable_data<T>(ctx.GetPlace()));
    XPUType* dbias_ptr =
        (dy == NULL)
            ? reinterpret_cast<XPUType*>(NULL)
            : reinterpret_cast<XPUType*>(dbias->mutable_data<T>(ctx.GetPlace()));

    r = xpu::fused_dense_fusion_grad<XPUType, XPUType, XPUType, XPUType, XPUType, int16_t>(
            xpu_ctx, x_ptr, y_ptr, dout_ptr,
            dx_ptr, dy_ptr, m, n, k, transx, transy,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            x_dims[1], y_dims[1], n, 1.0f, 0.0f, dbias_ptr,
            act, gelu_in_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "fused_dense_fusion_grad");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    custom_fused_dense,
    /*ops::CustomFusedDenseXPUKernel<phi::XPUContext, float>,*/
    ops::CustomFusedDenseXPUKernel<phi::XPUContext, paddle::platform::float16>);

REGISTER_OP_XPU_KERNEL(
    custom_fused_dense_grad,
    /*ops::CustomFusedDenseXPUGradKernel<phi::XPUContext, float>,*/
    ops::CustomFusedDenseXPUGradKernel<phi::XPUContext,
                                       paddle::platform::float16>);
