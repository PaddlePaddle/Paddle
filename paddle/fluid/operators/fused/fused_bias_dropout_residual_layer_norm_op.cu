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

#include <cuda_fp16.h>

#include <cub/cub.cuh>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class FusedBiasDropoutResidualLnOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    using U = LayerNormParamType<T>;
    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    auto *bias = ctx.Input<phi::DenseTensor>("Bias");
    auto *residual = ctx.Input<phi::DenseTensor>("Residual");
    const float ln_epsilon = ctx.Attr<float>("ln_epsilon");
    auto *ln_scale = ctx.Input<phi::DenseTensor>("LnScale");
    auto *ln_bias = ctx.Input<phi::DenseTensor>("LnBias");
    auto *dropout_mask_out = ctx.Output<phi::DenseTensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Output<phi::DenseTensor>("BiasDropoutResidualOut");
    auto *ln_mean = ctx.Output<phi::DenseTensor>("LnMean");
    auto *ln_var = ctx.Output<phi::DenseTensor>("LnVariance");
    auto *y = ctx.Output<phi::DenseTensor>("Y");
    auto *x_data = input_x->data<T>();
    auto *bias_data = (bias == nullptr) ? nullptr : bias->data<T>();
    auto *residual_data = (residual == nullptr) ? nullptr : residual->data<T>();
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *ln_bias_data = (ln_bias == nullptr ? nullptr : ln_bias->data<U>());
    auto *bias_dropout_residual_out_data =
        dev_ctx.Alloc<T>(bias_dropout_residual_out,
                         bias_dropout_residual_out->numel() * sizeof(T));
    auto *ln_mean_data =
        dev_ctx.Alloc<U>(ln_mean, ln_mean->numel() * sizeof(U));
    auto *ln_var_data = dev_ctx.Alloc<U>(ln_var, ln_var->numel() * sizeof(U));
    auto *dropout_mask_out_data = dev_ctx.Alloc<uint8_t>(
        dropout_mask_out, dropout_mask_out->numel() * sizeof(uint8_t));
    auto *y_data = dev_ctx.Alloc<T>(y, y->numel() * sizeof(T));

    const auto input_x_dims = input_x->dims();
    int bsz_seq = 1;
    for (int i = 0; i < input_x_dims.size() - 1; i++) {
      bsz_seq *= input_x_dims[i];
    }
    int dim_embed = input_x_dims[input_x_dims.size() - 1];
    DropoutParam dropout_param(ctx, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx.cuda_device_context(),
        bsz_seq,
        dim_embed,
        dropout_param,
        ln_epsilon);
    // output = layernorm(residual + dropout(input + bias))
    fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
        ctx.cuda_device_context(),
        x_data,
        residual_data,
        bias_data,
        ln_scale_data,
        ln_bias_data,
        bias_dropout_residual_out_data,
        dropout_mask_out_data,
        y_data,
        ln_mean_data,
        ln_var_data);
  }
};

template <typename T>
class FusedBiasDropoutResidualLnGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    const float ln_epsilon = ctx.Attr<float>("ln_epsilon");
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto *d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    auto *ln_scale = ctx.Input<phi::DenseTensor>("LnScale");
    auto *dropout_mask_out = ctx.Input<phi::DenseTensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Input<phi::DenseTensor>("BiasDropoutResidualOut");
    auto *ln_mean = ctx.Input<phi::DenseTensor>("LnMean");
    auto *ln_var = ctx.Input<phi::DenseTensor>("LnVariance");
    auto *d_y_data = d_y->data<T>();
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *dropout_mask_out_data = dropout_mask_out->data<uint8_t>();
    auto *bias_dropout_residual_out_data = bias_dropout_residual_out->data<T>();
    auto *ln_mean_data = ln_mean->data<U>();
    auto *ln_var_data = ln_var->data<U>();

    auto *d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *d_residual =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Residual"));
    auto *d_bias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));
    auto *d_bias_dropout_residual_out = ctx.Output<phi::DenseTensor>(
        framework::GradVarName("BiasDropoutResidualOut"));
    auto *d_ln_scale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("LnScale"));
    auto *d_ln_bias =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("LnBias"));
    auto *d_x_data = dev_ctx.Alloc<T>(d_x, d_x->numel() * sizeof(T));
    auto *d_residual_data =
        dev_ctx.Alloc<T>(d_residual, d_residual->numel() * sizeof(T));
    auto *d_bias_dropout_residual_out_data =
        dev_ctx.Alloc<T>(d_bias_dropout_residual_out,
                         d_bias_dropout_residual_out->numel() * sizeof(T));
    auto *d_bias_data =
        (d_bias == nullptr
             ? nullptr
             : dev_ctx.Alloc<T>(d_bias, d_bias->numel() * sizeof(T)));
    auto *d_ln_scale_data =
        (d_ln_scale == nullptr
             ? nullptr
             : dev_ctx.Alloc<U>(d_ln_scale, d_ln_scale->numel() * sizeof(U)));
    auto *d_ln_bias_data =
        (d_ln_bias == nullptr
             ? nullptr
             : dev_ctx.Alloc<U>(d_ln_bias, d_ln_bias->numel() * sizeof(U)));

    const auto input_x_dims = d_y->dims();
    int bsz_seq = 1;
    for (int i = 0; i < input_x_dims.size() - 1; i++) {
      bsz_seq *= input_x_dims[i];
    }
    int dim_embed = input_x_dims[input_x_dims.size() - 1];
    DropoutParam dropout_param(ctx, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx.cuda_device_context(),
        bsz_seq,
        dim_embed,
        dropout_param,
        ln_epsilon);
    fused_dropout_layernorm_helper.LayernormResidualDropoutBiasGrad(
        ctx.cuda_device_context(),
        d_y_data,
        bias_dropout_residual_out_data,
        dropout_mask_out_data,
        ln_scale_data,
        ln_mean_data,
        ln_var_data,
        d_bias_dropout_residual_out_data,
        d_ln_scale_data,
        d_ln_bias_data,
        d_x_data,
        d_bias_data,
        d_residual_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_bias_dropout_residual_layer_norm,
                        ops::FusedBiasDropoutResidualLnOpKernel<float>,
                        ops::FusedBiasDropoutResidualLnOpKernel<double>,
                        ops::FusedBiasDropoutResidualLnOpKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    fused_bias_dropout_residual_layer_norm_grad,
    ops::FusedBiasDropoutResidualLnGradKernel<float>,
    ops::FusedBiasDropoutResidualLnGradKernel<double>,
    ops::FusedBiasDropoutResidualLnGradKernel<plat::float16>);
