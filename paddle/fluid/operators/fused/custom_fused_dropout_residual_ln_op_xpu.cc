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
class CustomFusedDropoutResidualLnXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
  auto& dev_ctx = ctx.template device_context<phi::XPUContext>();

  const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
  const phi::DenseTensor* residual = ctx.Input<phi::DenseTensor>("Residual");
  const phi::DenseTensor* ln_scale = ctx.Input<phi::DenseTensor>("LnScale");
  const phi::DenseTensor* ln_bias = ctx.Input<phi::DenseTensor>("LnBias");
  
  phi::DenseTensor* final_out = ctx.Output<phi::DenseTensor>("Out");
  phi::DenseTensor* dropout_mask_out = ctx.Output<phi::DenseTensor>("DropoutMask");
  phi::DenseTensor* ln_mean = ctx.Output<phi::DenseTensor>("LnMean");
  phi::DenseTensor* ln_var = ctx.Output<phi::DenseTensor>("LnVar");
  phi::DenseTensor* dropout_residual_out = ctx.Output<phi::DenseTensor>("DropoutResidualOut");

  auto ln_epsilon = ctx.Attr<float>("ln_epsilon");
  auto is_test = ctx.Attr<bool>("is_test");
//   auto fix_seed = ctx.Attr<bool>("fix_seed");
  auto seed_val = ctx.Attr<int>("seed_val");
  auto is_upscale_in_train = ctx.Attr<bool>("is_upscale_in_train");
  auto dropout_rate = ctx.Attr<float>("dropout_rate");

  auto* xpu_ctx = dev_ctx.x_context();
  // inputs
  const auto &x_dims = x->dims();
  int x_m = 1;
  for (int i = 0; i < x_dims.size() - 1; i++) {
    x_m *= x_dims[i];
  }
  int x_n = x_dims[x_dims.size() - 1];

//   auto matrix_dim = phi::flatten_to_2d(x_dims, 0);
//   int x_m = static_cast<int>(matrix_dim[0]);
//   int x_n = static_cast<int>(matrix_dim[1]);

  printf("===> custom_fused_dropout_residual_ln, m: %d, n: %d\n", x_m, x_n);

//   // outputs
//   final_out.Resize(x_dims);
//   dropout_mask_out.Resize(x_dims);
//   dropout_residual_out.Resize(x_dims);
//   ln_mean.Resize({x_m});
//   ln_var.Resize({x_m});

  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
  const XPUType* res_ptr = reinterpret_cast<const XPUType*>(residual->data<T>());
  const XPUType* ln_scale_ptr = reinterpret_cast<const XPUType*>(ln_scale->data<T>());
  const XPUType* ln_bias_ptr = reinterpret_cast<const XPUType*>(ln_bias->data<T>());

  XPUType* out_ptr = reinterpret_cast<XPUType*>(final_out->mutable_data<T>(ctx.GetPlace()));

  XPUType* dropout_out_ptr = reinterpret_cast<XPUType*>(dropout_residual_out->mutable_data<T>(ctx.GetPlace()));
  XPUType* dropout_mask_out_ptr = reinterpret_cast<XPUType*>(dropout_mask_out->mutable_data<T>(ctx.GetPlace()));
  XPUType* ln_mean_ptr = reinterpret_cast<XPUType*>(ln_mean->mutable_data<T>(ctx.GetPlace()));
  XPUType* ln_var_ptr = reinterpret_cast<XPUType*>(ln_var->mutable_data<T>(ctx.GetPlace()));


  xpu::DropoutAddLayernormParam dropout_param = {is_test, is_upscale_in_train, dropout_rate, seed_val, true, ln_epsilon, x_m, x_n};
  
  int r = xpu::dropout_add_layernorm(xpu_ctx,
        x_ptr, res_ptr, ln_scale_ptr, ln_bias_ptr,
        dropout_out_ptr, dropout_mask_out_ptr, out_ptr, ln_mean_ptr, ln_var_ptr, (const xpu::DropoutAddLayernormParam)dropout_param);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_add_layernorm");
  }
};

template <typename DeviceContext, typename T>
class CustomFusedDropoutResidualLnXPUGradKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if 0
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();

    const phi::DenseTensor* ln_scale = *ctx.Input<f::Tensor>("LnScale");
    const phi::DenseTensor* dropout_mask_out = *ctx.Input<f::Tensor>("DropoutMask");
    const phi::DenseTensor* ln_mean = *ctx.Input<f::Tensor>("LnMean");
    const phi::DenseTensor* ln_var = *ctx.Input<f::Tensor>("LnVar");
    const phi::DenseTensor* dropout_residual_out =
        *ctx.Input<f::Tensor>("DropoutResidualOut");
    const phi::DenseTensor* grad_out = *ctx.Input<f::Tensor>(f::GradVarName("Out"));
    auto &grad_x = *ctx.Output<f::Tensor>(f::GradVarName("X"));
    auto &grad_residual = *ctx.Output<f::Tensor>(f::GradVarName("Residual"));
    auto &grad_ln_scale = *ctx.Output<f::Tensor>(f::GradVarName("LnScale"));
    auto &grad_ln_bias = *ctx.Output<f::Tensor>(f::GradVarName("LnBias"));
    f::Tensor grad_dropout_residual_out;

    auto ln_epsilon = ctx.Attr<float>("ln_epsilon");
    auto is_test = ctx.Attr<bool>("is_test");
    auto fix_seed = ctx.Attr<bool>("fix_seed");
    auto seed_val = ctx.Attr<int>("seed_val");
    auto is_upscale_in_train = ctx.Attr<bool>("is_upscale_in_train");
    auto dropout_rate = ctx.Attr<float>("dropout_rate");
#endif

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    custom_fused_dropout_residual_ln,
    ops::CustomFusedDropoutResidualLnXPUKernel<phi::XPUContext, float>);
//ops::CustomFusedDropoutResidualLnXPUKernel<phi::XPUContext, paddle::platform::float16>

REGISTER_OP_XPU_KERNEL(
    custom_fused_dropout_residual_ln_grad,
    ops::CustomFusedDropoutResidualLnXPUGradKernel<phi::XPUContext, float>);
