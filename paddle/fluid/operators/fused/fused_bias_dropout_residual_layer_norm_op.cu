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

using Tensor = framework::Tensor;

template <typename T>
class FusedBiasDropoutResidualLnOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto *input_x = ctx.Input<Tensor>("X");
    auto *bias = ctx.Input<Tensor>("Bias");
    auto *residual = ctx.Input<Tensor>("Residual");
    const float ln_epsilon = ctx.Attr<float>("ln_epsilon");
    auto *ln_scale = ctx.Input<Tensor>("LnScale");
    auto *ln_bias = ctx.Input<Tensor>("LnBias");
    auto *dropout_mask_out = ctx.Output<Tensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Output<Tensor>("BiasDropoutResidualOut");
    auto *ln_mean = ctx.Output<Tensor>("LnMean");
    auto *ln_var = ctx.Output<Tensor>("LnVariance");
    auto *y = ctx.Output<Tensor>("Y");
    auto *x_data = input_x->data<T>();
    auto *bias_data = (bias == nullptr) ? nullptr : bias->data<T>();
    auto *residual_data = (residual == nullptr) ? nullptr : residual->data<T>();
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *ln_bias_data = (ln_bias == nullptr ? nullptr : ln_bias->data<U>());
    auto *bias_dropout_residual_out_data =
        bias_dropout_residual_out->mutable_data<T>(ctx.GetPlace());
    auto *ln_mean_data = ln_mean->mutable_data<U>(ctx.GetPlace());
    auto *ln_var_data = ln_var->mutable_data<U>(ctx.GetPlace());
    auto *dropout_mask_out_data =
        dropout_mask_out->mutable_data<uint8_t>(ctx.GetPlace());
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());

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

    auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto *ln_scale = ctx.Input<Tensor>("LnScale");
    auto *dropout_mask_out = ctx.Input<Tensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Input<Tensor>("BiasDropoutResidualOut");
    auto *ln_mean = ctx.Input<Tensor>("LnMean");
    auto *ln_var = ctx.Input<Tensor>("LnVariance");
    auto *d_y_data = d_y->data<T>();
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *dropout_mask_out_data = dropout_mask_out->data<uint8_t>();
    auto *bias_dropout_residual_out_data = bias_dropout_residual_out->data<T>();
    auto *ln_mean_data = ln_mean->data<U>();
    auto *ln_var_data = ln_var->data<U>();

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_residual = ctx.Output<Tensor>(framework::GradVarName("Residual"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    auto *d_bias_dropout_residual_out =
        ctx.Output<Tensor>(framework::GradVarName("BiasDropoutResidualOut"));
    auto *d_ln_scale = ctx.Output<Tensor>(framework::GradVarName("LnScale"));
    auto *d_ln_bias = ctx.Output<Tensor>(framework::GradVarName("LnBias"));
    auto *d_x_data = d_x->mutable_data<T>(ctx.GetPlace());
    auto *d_residual_data = d_residual->mutable_data<T>(ctx.GetPlace());
    auto *d_bias_dropout_residual_out_data =
        d_bias_dropout_residual_out->mutable_data<T>(ctx.GetPlace());
    auto *d_bias_data =
        (d_bias == nullptr ? nullptr : d_bias->mutable_data<T>(ctx.GetPlace()));
    auto *d_ln_scale_data =
        (d_ln_scale == nullptr ? nullptr
                               : d_ln_scale->mutable_data<U>(ctx.GetPlace()));
    auto *d_ln_bias_data =
        (d_ln_bias == nullptr ? nullptr
                              : d_ln_bias->mutable_data<U>(ctx.GetPlace()));

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
