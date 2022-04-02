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

#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_XPU

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class DropoutXPUKernel : public framework::OpKernel<T> {
  using XPUTyp = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");
    auto dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    auto& dev_ctx = context.template device_context<DeviceContext>();

    auto* seed =
        context.HasInput("Seed") ? context.Input<Tensor>("Seed") : nullptr;

    int is_upscale = (dropout_implementation == "upscale_in_train");

    if (!context.Attr<bool>("is_test")) {
      int seed_data = 0;
      if (seed) {
        if (platform::is_xpu_place(seed->place())) {
          memory::Copy(platform::CPUPlace(), &seed_data, seed->place(),
                       seed->data<int>(), sizeof(int));
        } else {
          seed_data = *(seed->data<int>());
        }

      } else {
        seed_data =
            context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : 0;
      }

      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<T>(context.GetPlace());
      // Special case when dropout_prob is 1.0
      if (dropout_prob == 1.0f) {
        int r = xpu::constant(dev_ctx.x_context(),
                              reinterpret_cast<XPUTyp*>(y_data), y->numel(),
                              XPUTyp(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
        r = xpu::constant(dev_ctx.x_context(),
                          reinterpret_cast<XPUTyp*>(mask_data), mask->numel(),
                          XPUTyp(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
        return;
      }
      int r = xpu::dropout(dev_ctx.x_context(),
                           reinterpret_cast<const XPUTyp*>(x->data<T>()),
                           reinterpret_cast<XPUTyp*>(y->data<T>()),
                           reinterpret_cast<XPUTyp*>(mask_data), seed_data,
                           mask->numel(), is_upscale, dropout_prob);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout");
    } else {
      float scale =
          (is_upscale) ? (1.0) : (static_cast<float>(1.0f - dropout_prob));
      int r = xpu::scale(
          dev_ctx.x_context(), reinterpret_cast<const XPUTyp*>(x_data),
          reinterpret_cast<XPUTyp*>(y_data), x->numel(), false, scale, 0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
  }
};
template <typename DeviceContext, typename T>
class DropoutGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(!context.Attr<bool>("is_test"), true,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test is false"));
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    float dropout_prob = context.Attr<float>("dropout_prob");
    const T* mask_data = mask->data<T>();

    if (dropout_implementation != "upscale_in_train") {
      int r = xpu::mul(dev_ctx.x_context(),
                       reinterpret_cast<const XPUType*>(grad_y->data<T>()),
                       reinterpret_cast<const XPUType*>(mask_data),
                       reinterpret_cast<XPUType*>(grad_x->data<T>()),
                       grad_y->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
      return;
    }

    auto version = platform::get_xpu_version(context.GetPlace().GetDeviceId());
    if (version == phi::backends::xpu::XPUVersion::XPU1) {
      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      XPUType* mask_new = RAII_GUARD.alloc_l3_or_gm<XPUType>(mask->numel());
      float scale =
          (dropout_prob == 1.0f) ? (1.0f) : (1.0f / (1.0f - dropout_prob));
      int r = xpu::scale(dev_ctx.x_context(),
                         reinterpret_cast<const XPUType*>(mask->data<T>()),
                         reinterpret_cast<XPUType*>(mask_new), mask->numel(),
                         false, scale, 0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
      r = xpu::mul(dev_ctx.x_context(),
                   reinterpret_cast<const XPUType*>(grad_y->data<T>()),
                   reinterpret_cast<const XPUType*>(mask_new),
                   reinterpret_cast<XPUType*>(grad_x->data<T>()),
                   grad_y->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
    } else {
      int r =
          xpu::dropout_grad(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(mask->data<T>()),
                            reinterpret_cast<const XPUType*>(grad_y->data<T>()),
                            reinterpret_cast<XPUType*>(grad_x->data<T>()),
                            dropout_prob, grad_y->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_grad");
    }
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(
    dropout, ops::DropoutXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::DropoutXPUKernel<paddle::platform::XPUDeviceContext, plat::float16>);
REGISTER_OP_XPU_KERNEL(
    dropout_grad,
    ops::DropoutGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::DropoutGradXPUKernel<paddle::platform::XPUDeviceContext,
                              plat::float16>);
#endif
