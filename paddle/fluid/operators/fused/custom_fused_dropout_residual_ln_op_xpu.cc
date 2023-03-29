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
    phi::DenseTensor* dropout_mask_out =
        ctx.Output<phi::DenseTensor>("DropoutMask");
    phi::DenseTensor* ln_mean = ctx.Output<phi::DenseTensor>("LnMean");
    phi::DenseTensor* ln_var = ctx.Output<phi::DenseTensor>("LnVar");
    phi::DenseTensor* dropout_residual_out =
        ctx.Output<phi::DenseTensor>("DropoutResidualOut");

    if (ln_mean->dtype() == phi::DataType::FLOAT32) {
      VLOG(1) << "layer_norm fusion, FLOAT32";
    } else if (ln_mean->dtype() == phi::DataType::FLOAT16) {
      VLOG(1) << "layer_norm fusion, FLOAT16\n";
    }

    // input
    // paddle::experimental::CppTypeToDataType<T>::Type()
    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* res_ptr =
        reinterpret_cast<const XPUType*>(residual->data<T>());
    const XPUType* ln_scale_ptr =
        reinterpret_cast<const XPUType*>(ln_scale->data<T>());
    const XPUType* ln_bias_ptr =
        reinterpret_cast<const XPUType*>(ln_bias->data<T>());

    // output
    XPUType* out_ptr =
        reinterpret_cast<XPUType*>(final_out->mutable_data<T>(ctx.GetPlace()));
    XPUType* dropout_mask_out_ptr = reinterpret_cast<XPUType*>(
        dropout_mask_out->mutable_data<T>(ctx.GetPlace()));
    //   XPUType* dropout_mask_out_ptr =
    //   reinterpret_cast<XPUType*>(dropout_mask_out->mutable_data<uint8_t>(ctx.GetPlace()));
    float* ln_mean_ptr =
        reinterpret_cast<float*>(ln_mean->mutable_data<float>(ctx.GetPlace()));
    float* ln_var_ptr =
        reinterpret_cast<float*>(ln_var->mutable_data<float>(ctx.GetPlace()));
    XPUType* dropout_out_ptr = reinterpret_cast<XPUType*>(
        dropout_residual_out->mutable_data<T>(ctx.GetPlace()));

    VLOG(1) << "==> CustomFusedDropoutResidualLnXPUKernel";

    auto ln_epsilon = ctx.Attr<float>("ln_epsilon");
    auto is_test = ctx.Attr<bool>("is_test");
    //   auto fix_seed = ctx.Attr<bool>("fix_seed");
    auto seed_val = ctx.Attr<int>("seed_val");
    auto is_upscale_in_train = ctx.Attr<bool>("is_upscale_in_train");
    auto dropout_rate = ctx.Attr<float>("dropout_rate");

    auto* xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    // inputs
    const auto& x_dims = x->dims();
    int x_m = 1;
    for (int i = 0; i < x_dims.size() - 1; i++) {
      x_m *= x_dims[i];
    }
    int x_n = x_dims[x_dims.size() - 1];

    xpu::DropoutAddLayernormParam dropout_param = {is_test,
                                                   is_upscale_in_train,
                                                   dropout_rate,
                                                   seed_val,
                                                   true,
                                                   ln_epsilon,
                                                   x_m,
                                                   x_n};

    const float* ln_scale_fp32_ptr;
    const float* ln_bias_fp32_ptr;
    if (x->dtype() == phi::DataType::FLOAT32) {
      ln_scale_fp32_ptr = reinterpret_cast<const float*>(ln_scale_ptr);
      ln_bias_fp32_ptr = reinterpret_cast<const float*>(ln_bias_ptr);
    } else if (x->dtype() == phi::DataType::FLOAT16) {
      float* ln_scale_tmp = RAII_GUARD.alloc_l3_or_gm<float>(ln_scale->numel());
      float* ln_bias_tmp = RAII_GUARD.alloc_l3_or_gm<float>(ln_bias->numel());
      int r = xpu::cast<XPUType, float>(
          xpu_ctx, ln_scale_ptr, ln_scale_tmp, ln_scale->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      r = xpu::cast<XPUType, float>(
          xpu_ctx, ln_bias_ptr, ln_bias_tmp, ln_bias->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      ln_scale_fp32_ptr = ln_scale_tmp;
      ln_bias_fp32_ptr = ln_bias_tmp;
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("custom_fused_dropout_residual_ln lnscale "
                                     "ln_bias only support fp32, fp16"));
    }
    int r = xpu::dropout_add_layernorm<XPUType>(
        xpu_ctx,
        x_ptr,
        res_ptr,
        ln_scale_fp32_ptr,
        ln_bias_fp32_ptr,
        dropout_out_ptr,
        (bit16_t *)dropout_mask_out_ptr,
        out_ptr,
        ln_mean_ptr,
        ln_var_ptr,
        (const xpu::DropoutAddLayernormParam)dropout_param);

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_add_layernorm");
  }
};

template <typename DeviceContext, typename T>
class CustomFusedDropoutResidualLnXPUGradKernel
    : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // PADDLE_THROW(platform::errors::Unimplemented(
    //     "The custom_fused_dropout_residual_ln_grad operator does not support
    //     XPU yet."));
    VLOG(1) << "==> CustomFusedDropoutResidualLnXPUGradKernel";

    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();

    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    // const phi::DenseTensor* residual =
    //     ctx.Input<phi::DenseTensor>("Residual");
    const phi::DenseTensor* ln_scale = ctx.Input<phi::DenseTensor>("LnScale");
    // const phi::DenseTensor* ln_bias = ctx.Input<phi::DenseTensor>("LnBias");

    const phi::DenseTensor* dropout_mask_out =
        ctx.Input<phi::DenseTensor>("DropoutMask");
    const phi::DenseTensor* ln_mean = ctx.Input<phi::DenseTensor>("LnMean");
    const phi::DenseTensor* ln_var = ctx.Input<phi::DenseTensor>("LnVar");
    const phi::DenseTensor* dropout_residual_out =
        ctx.Input<phi::DenseTensor>("DropoutResidualOut");
    const phi::DenseTensor* grad_out =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    // output
    phi::DenseTensor* grad_x =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    phi::DenseTensor* grad_residual =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Residual"));
    phi::DenseTensor* grad_ln_scale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("LnScale"));
    phi::DenseTensor* grad_ln_bias =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("LnBias"));

    // input
    // const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
    // const XPUType* res_ptr =
    //         reinterpret_cast<const XPUType*>(residual->data<T>());
    const XPUType* ln_scale_ptr =
        reinterpret_cast<const XPUType*>(ln_scale->data<T>());
    // const XPUType* ln_bias_ptr =
    //     reinterpret_cast<const XPUType*>(ln_bias->data<T>());
    const XPUType* dropout_mask_ptr =
        reinterpret_cast<const XPUType*>(dropout_mask_out->data<T>());
    const float* ln_mean_ptr =
        reinterpret_cast<const float*>(ln_mean->data<float>());
    const float* ln_var_ptr =
        reinterpret_cast<const float*>(ln_var->data<float>());
    const XPUType* dropout_out_ptr =
        reinterpret_cast<const XPUType*>(dropout_residual_out->data<T>());
    const XPUType* grad_out_ptr =
        reinterpret_cast<const XPUType*>(grad_out->data<T>());

    // output
    XPUType* dx_ptr =
        reinterpret_cast<XPUType*>(grad_x->mutable_data<T>(ctx.GetPlace()));
    XPUType* d_residual_ptr = reinterpret_cast<XPUType*>(
        grad_residual->mutable_data<T>(ctx.GetPlace()));
    XPUType* d_ln_scale_ptr = reinterpret_cast<XPUType*>(
        grad_ln_scale->mutable_data<T>(ctx.GetPlace()));
    XPUType* d_ln_bias_ptr = reinterpret_cast<XPUType*>(
        grad_ln_bias->mutable_data<T>(ctx.GetPlace()));

    auto ln_epsilon = ctx.Attr<float>("ln_epsilon");
    auto is_test = ctx.Attr<bool>("is_test");
    // auto fix_seed = ctx.Attr<bool>("fix_seed");
    auto seed_val = ctx.Attr<int>("seed_val");
    auto is_upscale_in_train = ctx.Attr<bool>("is_upscale_in_train");
    auto dropout_rate = ctx.Attr<float>("dropout_rate");

    auto* xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    // inputs
    const auto& x_dims = x->dims();
    int x_m = 1;
    for (int i = 0; i < x_dims.size() - 1; i++) {
      x_m *= x_dims[i];
    }
    int x_n = x_dims[x_dims.size() - 1];

    xpu::DropoutAddLayernormParam dropout_param = {is_test,
                                                   is_upscale_in_train,
                                                   dropout_rate,
                                                   seed_val,
                                                   true,
                                                   ln_epsilon,
                                                   x_m,
                                                   x_n};

    // (void)dev_ctx;
    // (void)dx_ptr;
    // (void)d_residual_ptr;
    // (void)d_ln_scale_ptr;
    // (void)d_ln_bias_ptr;

    const float* ln_scale_fp32_ptr;
    float* ln_dscale_fp32_ptr;
    float* ln_dbias_fp32_ptr;
    if (x->dtype() == phi::DataType::FLOAT32) {
      ln_scale_fp32_ptr = reinterpret_cast<const float*>(ln_scale_ptr);
      ln_dscale_fp32_ptr = reinterpret_cast<float*>(d_ln_scale_ptr);
      ln_dbias_fp32_ptr = reinterpret_cast<float*>(d_ln_bias_ptr);
    } else if (x->dtype() == phi::DataType::FLOAT16) {
      float* ln_scale_tmp = RAII_GUARD.alloc_l3_or_gm<float>(ln_scale->numel());
      ln_dscale_fp32_ptr =
          RAII_GUARD.alloc_l3_or_gm<float>(grad_ln_scale->numel());
      ln_dbias_fp32_ptr =
          RAII_GUARD.alloc_l3_or_gm<float>(grad_ln_bias->numel());
      int r = xpu::cast<XPUType, float>(
          xpu_ctx, ln_scale_ptr, ln_scale_tmp, ln_scale->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      ln_scale_fp32_ptr = ln_scale_tmp;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "custom_fused_dropout_residual_ln_grad lnscale ln_bias only support "
          "fp32, fp16"));
    }

    dropout_add_layernorm_grad(
        xpu_ctx,
        dropout_out_ptr,
        (bit16_t *)dropout_mask_ptr,
        grad_out_ptr,
        dx_ptr,
        d_residual_ptr,
        ln_scale_fp32_ptr,
        ln_mean_ptr,
        ln_var_ptr,
        ln_dscale_fp32_ptr,
        ln_dbias_fp32_ptr,
        (const xpu::DropoutAddLayernormParam)dropout_param);

    // ret = api::clip<float>(ctx, dx_tmp, dx_tmp, xm * n, -65504.0, 65504.0);
    if (x->dtype() == phi::DataType::FLOAT16) {
      int r = xpu::clip<float>(xpu_ctx,
                               ln_dscale_fp32_ptr,
                               ln_dscale_fp32_ptr,
                               grad_ln_scale->numel(),
                               -65504.0,
                               65504.0);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip");
      r = xpu::cast<float, XPUType>(
          xpu_ctx, ln_dscale_fp32_ptr, d_ln_scale_ptr, grad_ln_scale->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");

      r = xpu::clip<float>(xpu_ctx,
                           ln_dbias_fp32_ptr,
                           ln_dbias_fp32_ptr,
                           grad_ln_bias->numel(),
                           -65504.0,
                           65504.0);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip");
      r = xpu::cast<float, XPUType>(
          xpu_ctx, ln_dbias_fp32_ptr, d_ln_bias_ptr, grad_ln_bias->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    }

#if 0
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();

    const phi::DenseTensor* ln_scale =
        ctx.Input<phi::DenseTensor>("LnScale");
    const phi::DenseTensor* dropout_mask_out =
        ctx.Input<phi::DenseTensor>("DropoutMask");
    const phi::DenseTensor* ln_mean =
        ctx.Input<phi::DenseTensor>("LnMean");
    const phi::DenseTensor* ln_var = ctx.Input<phi::DenseTensor>("LnVar");
    const phi::DenseTensor* dropout_residual_out =
        ctx.Input<phi::DenseTensor>("Residual");
    const phi::DenseTensor* grad_out =
        ctx.Input<phi::DenseTensor>(f::GradVarName("Out"));

    phi::DenseTensor* grad_x =
        ctx.Output<phi::DenseTensor>(f::GradVarName("X"));
    phi::DenseTensor* grad_residual =
        ctx.Output<phi::DenseTensor>(f::GradVarName("Residual"));
    phi::DenseTensor* grad_ln_scale =
        ctx.Output<phi::DenseTensor>(f::GradVarName("LnScale"));
    phi::DenseTensor* grad_ln_bias =
        ctx.Output<phi::DenseTensor>(f::GradVarName("LnBias"));
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
    ops::CustomFusedDropoutResidualLnXPUKernel<phi::XPUContext, float>,
    ops::CustomFusedDropoutResidualLnXPUKernel<phi::XPUContext,
                                               paddle::platform::float16>);

REGISTER_OP_XPU_KERNEL(
    custom_fused_dropout_residual_ln_grad,
    ops::CustomFusedDropoutResidualLnXPUGradKernel<phi::XPUContext, float>,
    ops::CustomFusedDropoutResidualLnXPUGradKernel<phi::XPUContext,
                                                   paddle::platform::float16>);
