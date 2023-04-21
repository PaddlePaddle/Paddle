/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/fused/fused_attention_utils.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"
#include "paddle/phi/kernels/impl/matmul_grad_kernel_impl.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MatMul(const phi::GPUContext& dev_ctx,
            const phi::DenseTensor& a,
            const phi::DenseTensor& b,
            phi::DenseTensor* c) {
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  auto a_2d = phi::FoldInitDims(a);
  auto b_2d = phi::FoldInitDims(b);
  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a_2d.dims(), 0, false);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b_2d.dims(), 0, false);
  T alpha = static_cast<T>(1.0);
  blas.MatMul(a, mat_dim_a, b, mat_dim_b, alpha, c, T(0));
}

template <typename T, typename Context>
void FFN(const phi::GPUContext& dev_ctx,
         const phi::DenseTensor& x,
         const phi::DenseTensor& linear1_weight,
         const phi::DenseTensor* linear1_bias,
         const phi::DenseTensor& linear2_weight,
         const phi::DenseTensor* linear2_bias,
         const phi::DenseTensor* ln1_scale,
         const phi::DenseTensor* ln1_bias,
         const phi::DenseTensor* ln2_scale,
         const phi::DenseTensor* ln2_bias,
         phi::DenseTensor* out,
         phi::DenseTensor* dropout1_mask,
         phi::DenseTensor* dropout2_mask,
         phi::DenseTensor* ln1_mean,
         phi::DenseTensor* ln1_variance,
         phi::DenseTensor* ln2_mean,
         phi::DenseTensor* ln2_variance,
         phi::DenseTensor* linear1_out,
         phi::DenseTensor* ln1_out,
         phi::DenseTensor* dropout1_out,
         phi::DenseTensor* dropout2_out,
         const int bsz_seq,
         const int d_model,
         const int dim_feedforward,
         const std::string& act_method,
         const bool pre_layer_norm,
         const float epsilon1,
         const float epsilon2,
         const bool add_residual,
         const int ring_id,
         const phi::fusion::DropoutParam& dropout_param1,
         const phi::fusion::DropoutParam& dropout_param2) {
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t> pre_layernorm_helper(
      bsz_seq, d_model, epsilon1);
  phi::fusion::FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
      dev_ctx, bsz_seq, dim_feedforward, dropout_param1);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      fused_dropout_layernorm_helper(
          dev_ctx, bsz_seq, d_model, dropout_param2, epsilon2);

  using U = phi::funcs::LayerNormParamType<T>;
  const phi::DenseTensor* in = &x;

  const U* ln1_scale_ptr =
      ln1_scale == nullptr ? nullptr : ln1_scale->data<U>();
  const U* ln1_bias_ptr = ln1_bias == nullptr ? nullptr : ln1_bias->data<U>();
  const U* ln2_scale_ptr =
      ln2_scale == nullptr ? nullptr : ln2_scale->data<U>();
  const U* ln2_bias_ptr = ln2_bias == nullptr ? nullptr : ln2_bias->data<U>();
  const T* linear1_bias_ptr =
      linear1_bias == nullptr ? nullptr : linear1_bias->data<T>();
  const T* linear2_bias_ptr =
      linear2_bias == nullptr ? nullptr : linear2_bias->data<T>();

  if (pre_layer_norm) {
    pre_layernorm_helper.LayerNorm(dev_ctx,
                                   x.data<T>(),
                                   ln1_scale_ptr,
                                   ln1_bias_ptr,
                                   ln1_out->data<T>(),
                                   ln1_mean->data<U>(),
                                   ln1_variance->data<U>());
    in = ln1_out;
  }
  MatMul<T, Context>(dev_ctx, *in, linear1_weight, linear1_out);
  fused_act_dropout_helper.DropoutActBias(dev_ctx,
                                          linear1_out->data<T>(),
                                          linear1_bias_ptr,
                                          act_method,
                                          dropout1_out->data<T>(),
                                          dropout1_mask->data<uint8_t>());
  phi::DenseTensor linear2_out;
  linear2_out.Resize({bsz_seq, d_model});
  dev_ctx.template Alloc<T>(&linear2_out, linear2_out.numel() * sizeof(T));
  MatMul<T, Context>(dev_ctx, *dropout1_out, linear2_weight, &linear2_out);

  // tensor model parallel
  phi::fusion::AllReduce<T>(linear2_out, ring_id, dev_ctx);

  const T* residual_ptr = add_residual ? x.data<T>() : nullptr;
  if (!pre_layer_norm) {
    // TODO(Xreki): support post layer_norm case when add_residual is false.
    PADDLE_ENFORCE_EQ(add_residual,
                      true,
                      phi::errors::InvalidArgument(
                          "Attribute add_residual is expected to be true "
                          "when pre_layer_norm is false."));

    fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
        dev_ctx,
        linear2_out.data<T>(),
        residual_ptr,
        linear2_bias_ptr,
        ln2_scale_ptr,
        ln2_bias_ptr,
        dropout2_out->data<T>(),
        dropout2_mask->data<uint8_t>(),
        out->data<T>(),
        ln2_mean->data<U>(),
        ln2_variance->data<U>());
  } else {
    fused_dropout_layernorm_helper.ResidualDropoutBias(
        dev_ctx,
        linear2_out.data<T>(),
        residual_ptr,
        linear2_bias_ptr,
        out->data<T>(),
        dropout2_mask->data<uint8_t>());
  }
}

template <typename T, typename Context>
void FusedFeedForwardKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const paddle::optional<DenseTensor>& dropout1_seed,
                            const paddle::optional<DenseTensor>& dropout2_seed,
                            const DenseTensor& linear1_weight,
                            const paddle::optional<DenseTensor>& linear1_bias,
                            const DenseTensor& linear2_weight,
                            const paddle::optional<DenseTensor>& linear2_bias,
                            const paddle::optional<DenseTensor>& ln1_scale,
                            const paddle::optional<DenseTensor>& ln1_bias,
                            const paddle::optional<DenseTensor>& ln2_scale,
                            const paddle::optional<DenseTensor>& ln2_bias,
                            bool pre_layer_norm,
                            float ln1_epsilon,
                            float ln2_epsilon,
                            const std::string& act_method,
                            float dropout1_prob,
                            float dropout2_prob,
                            const std::string& dropout1_implementation,
                            const std::string& dropout2_implementation,
                            bool is_test,
                            bool dropout1_fix_seed,
                            bool dropout2_fix_seed,
                            int dropout1_seed_val,
                            int dropout2_seed_val,
                            bool add_residual,
                            int ring_id,
                            DenseTensor* out,
                            DenseTensor* dropout1_mask,
                            DenseTensor* dropout2_mask,
                            DenseTensor* ln1_mean,
                            DenseTensor* ln1_variance,
                            DenseTensor* ln2_mean,
                            DenseTensor* ln2_variance,
                            DenseTensor* linear1_out,
                            DenseTensor* ln1_out,
                            DenseTensor* dropout1_out,
                            DenseTensor* dropout2_out) {
  auto* x_ptr = &x;
  auto* linear1_weight_ptr = &linear1_weight;
  auto* linear1_bias_ptr = linear1_bias.get_ptr();
  auto* linear2_weight_ptr = &linear2_weight;
  auto* linear2_bias_ptr = linear2_bias.get_ptr();

  auto* ln1_scale_ptr = pre_layer_norm ? ln1_scale.get_ptr() : nullptr;
  auto* ln1_bias_ptr = pre_layer_norm ? ln1_bias.get_ptr() : nullptr;
  auto* ln2_scale_ptr = !pre_layer_norm ? ln2_scale.get_ptr() : nullptr;
  auto* ln2_bias_ptr = !pre_layer_norm ? ln2_bias.get_ptr() : nullptr;

  if (!pre_layer_norm) {
    ln1_mean = nullptr;
    ln1_variance = nullptr;
    ln1_out = nullptr;
  } else {
    ln2_mean = nullptr;
    ln2_variance = nullptr;
  }

  bool is_upscale_in_train1 = dropout1_implementation == "upscale_in_train";
  bool is_upscale_in_train2 = dropout2_implementation == "upscale_in_train";
  auto* dropout1_seed_ptr = dropout1_seed.get_ptr();
  auto* dropout2_seed_ptr = dropout2_seed.get_ptr();

  phi::fusion::DropoutParam dropout_param1(dropout1_fix_seed,
                                           0,
                                           is_test,
                                           is_upscale_in_train1,
                                           dropout1_prob,
                                           dropout1_seed_ptr,
                                           dropout1_seed_val);
  phi::fusion::DropoutParam dropout_param2(dropout2_fix_seed,
                                           0,
                                           is_test,
                                           is_upscale_in_train2,
                                           dropout2_prob,
                                           dropout2_seed_ptr,
                                           dropout2_seed_val);

  using U = phi::funcs::LayerNormParamType<T>;
  dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));
  dev_ctx.template Alloc<uint8_t>(dropout1_mask,
                                  dropout1_mask->numel() * sizeof(uint8_t));
  dev_ctx.template Alloc<uint8_t>(dropout2_mask,
                                  dropout2_mask->numel() * sizeof(uint8_t));
  if (pre_layer_norm) {
    dev_ctx.template Alloc<U>(ln1_mean, ln1_mean->numel() * sizeof(U));
    dev_ctx.template Alloc<U>(ln1_variance, ln1_variance->numel() * sizeof(U));
    dev_ctx.template Alloc<T>(ln1_out, ln1_out->numel() * sizeof(T));
  } else {
    dev_ctx.template Alloc<U>(ln2_mean, ln2_mean->numel() * sizeof(U));
    dev_ctx.template Alloc<U>(ln2_variance, ln2_variance->numel() * sizeof(U));
  }

  dev_ctx.template Alloc<T>(linear1_out, linear1_out->numel() * sizeof(T));
  dev_ctx.template Alloc<T>(dropout1_out, dropout1_out->numel() * sizeof(T));
  dev_ctx.template Alloc<T>(dropout2_out, dropout2_out->numel() * sizeof(T));

  auto x_dim = x_ptr->dims();
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
      phi::RowMatrixFromVector(x_dim), 0, false);

  auto dim = linear1_weight_ptr->dims();
  int d_model = dim[0];
  int dim_feedforward = dim[dim.size() - 1];
  int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;

  phi::fusion::FFN<T, Context>(dev_ctx,
                               x,
                               linear1_weight,
                               linear1_bias_ptr,
                               linear2_weight,
                               linear2_bias_ptr,
                               ln1_scale_ptr,
                               ln1_bias_ptr,
                               ln2_scale_ptr,
                               ln2_bias_ptr,
                               out,
                               dropout1_mask,
                               dropout2_mask,
                               ln1_mean,
                               ln1_variance,
                               ln2_mean,
                               ln2_variance,
                               linear1_out,
                               ln1_out,
                               dropout1_out,
                               dropout2_out,
                               bsz_seq,
                               d_model,
                               dim_feedforward,
                               act_method,
                               pre_layer_norm,
                               ln1_epsilon,
                               ln2_epsilon,
                               add_residual,
                               ring_id,
                               dropout_param1,
                               dropout_param2);
}

template <typename T, typename Context>
void MatMulGrad(const phi::GPUContext& dev_ctx,
                const phi::DenseTensor& d_out,
                const phi::DenseTensor& a,
                const phi::DenseTensor& b,
                phi::DenseTensor* d_a,
                phi::DenseTensor* d_b) {
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  auto a_2d = phi::FoldInitDims(a);
  auto b_2d = phi::FoldInitDims(b);
  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a_2d.dims(), 0, true);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b_2d.dims(), 0, true);
  auto mat_dim_dout =
      phi::funcs::CreateMatrixDescriptor(d_out.dims(), 0, false);
  T alpha = static_cast<T>(1.0);
  blas.MatMul(d_out, mat_dim_dout, b, mat_dim_b, alpha, d_a, T(0));
  blas.MatMul(a, mat_dim_a, d_out, mat_dim_dout, alpha, d_b, T(0));
}

template <typename T, typename Context>
void FFNGrad(const phi::GPUContext& dev_ctx,
             const phi::DenseTensor& d_out,
             const phi::DenseTensor& x,
             const phi::DenseTensor& dropout1_mask,
             const phi::DenseTensor& dropout2_mask,
             const phi::DenseTensor& linear1_out,
             const phi::DenseTensor* ln1_out,
             const phi::DenseTensor& dropout1_out,
             const phi::DenseTensor* dropout2_out,
             const phi::DenseTensor& linear1_weight,
             const phi::DenseTensor* linear1_bias,
             const phi::DenseTensor& linear2_weight,
             const phi::DenseTensor* ln1_gamma,
             const phi::DenseTensor* ln1_beta,
             const phi::DenseTensor* ln1_mean,
             const phi::DenseTensor* ln1_variance,
             const phi::DenseTensor* ln2_gamma,
             const phi::DenseTensor* ln2_beta,
             const phi::DenseTensor* ln2_mean,
             const phi::DenseTensor* ln2_variance,
             phi::DenseTensor* d_x,
             phi::DenseTensor* d_linear1_weight,
             phi::DenseTensor* d_linear1_bias,
             phi::DenseTensor* d_linear2_weight,
             phi::DenseTensor* d_linear2_bias,
             phi::DenseTensor* d_ln1_gamma,
             phi::DenseTensor* d_ln1_beta,
             phi::DenseTensor* d_ln2_gamma,
             phi::DenseTensor* d_ln2_beta,
             const int bsz_seq,
             const int d_model,
             const int dim_feedforward,
             const DropoutParam& dropout_param1,
             const DropoutParam& dropout_param2,
             const std::string& act_method,
             const bool pre_layer_norm,
             const float epsilon1,
             const float epsilon2,
             const bool add_residual,
             const int ring_id) {
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t> pre_layernorm_helper(
      bsz_seq, d_model, epsilon1);
  phi::fusion::FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
      dev_ctx, bsz_seq, dim_feedforward, dropout_param1);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      fused_dropout_layernorm_helper(
          dev_ctx, bsz_seq, d_model, dropout_param2, epsilon2);

  using U = phi::funcs::LayerNormParamType<T>;
  const U* ln1_gamma_ptr =
      ln1_gamma == nullptr ? nullptr : ln1_gamma->data<U>();
  const U* ln1_beta_ptr = ln1_beta == nullptr ? nullptr : ln1_beta->data<U>();
  const U* ln2_gamma_ptr =
      ln2_gamma == nullptr ? nullptr : ln2_gamma->data<U>();
  const U* ln2_beta_ptr = ln2_beta == nullptr ? nullptr : ln2_beta->data<U>();
  const T* linear1_bias_ptr =
      linear1_bias == nullptr ? nullptr : linear1_bias->data<T>();
  T* d_linear1_bias_ptr =
      d_linear1_bias == nullptr ? nullptr : d_linear1_bias->data<T>();
  T* d_linear2_bias_ptr =
      d_linear2_bias == nullptr ? nullptr : d_linear2_bias->data<T>();
  U* d_ln1_gamma_ptr =
      d_ln1_gamma == nullptr ? nullptr : d_ln1_gamma->data<U>();
  U* d_ln1_beta_ptr = d_ln1_beta == nullptr ? nullptr : d_ln1_beta->data<U>();
  U* d_ln2_gamma_ptr =
      d_ln2_gamma == nullptr ? nullptr : d_ln2_gamma->data<U>();
  U* d_ln2_beta_ptr = d_ln2_beta == nullptr ? nullptr : d_ln2_beta->data<U>();

  phi::DenseTensor d_linear2_out, d_dropout2_out, d_residual;
  d_linear2_out.Resize({bsz_seq, d_model});
  dev_ctx.template Alloc<T>(&d_linear2_out, d_linear2_out.numel() * sizeof(T));
  d_dropout2_out.Resize({bsz_seq, d_model});
  dev_ctx.template Alloc<T>(&d_dropout2_out,
                            d_dropout2_out.numel() * sizeof(T));

  T* d_residual_ptr = nullptr;
  if (add_residual) {
    d_residual.Resize(d_x->dims());
    d_residual_ptr =
        dev_ctx.template Alloc<T>(&d_residual, d_residual.numel() * sizeof(T));
  }
  if (pre_layer_norm) {
    fused_dropout_layernorm_helper.ResidualDropoutBiasGrad(
        dev_ctx,
        d_out.data<T>(),
        dropout2_mask.data<uint8_t>(),
        d_linear2_out.data<T>(),
        d_residual_ptr,
        d_linear2_bias_ptr);
  } else {
    fused_dropout_layernorm_helper.LayernormResidualDropoutBiasGrad(
        dev_ctx,
        d_out.data<T>(),
        dropout2_out->data<T>(),
        dropout2_mask.data<uint8_t>(),
        ln2_gamma_ptr,
        ln2_mean->data<U>(),
        ln2_variance->data<U>(),
        d_dropout2_out.data<T>(),
        d_ln2_gamma_ptr,
        d_ln2_beta_ptr,
        d_linear2_out.data<T>(),
        d_linear2_bias_ptr,
        d_residual_ptr);
  }

  phi::DenseTensor d_dropout1_out;
  d_dropout1_out.Resize({bsz_seq, dim_feedforward});
  dev_ctx.template Alloc<T>(&d_dropout1_out,
                            d_dropout1_out.numel() * sizeof(T));
  MatMulGrad<T, Context>(dev_ctx,
                         d_linear2_out,
                         dropout1_out,
                         linear2_weight,
                         &d_dropout1_out,
                         d_linear2_weight);

  phi::DenseTensor d_linear1_out;
  d_linear1_out.Resize({bsz_seq, dim_feedforward});
  dev_ctx.template Alloc<T>(&d_linear1_out, d_linear1_out.numel() * sizeof(T));
  fused_act_dropout_helper.DropoutActBiasGrad(dev_ctx,
                                              d_dropout1_out.data<T>(),
                                              linear1_out.data<T>(),
                                              linear1_bias_ptr,
                                              dropout1_mask.data<uint8_t>(),
                                              d_linear1_out.data<T>(),
                                              d_linear1_bias_ptr,
                                              act_method);

  if (pre_layer_norm) {
    phi::DenseTensor d_ln1_out;
    d_ln1_out.Resize({bsz_seq, d_model});
    dev_ctx.template Alloc<T>(&d_ln1_out, d_ln1_out.numel() * sizeof(T));
    MatMulGrad<T, Context>(dev_ctx,
                           d_linear1_out,
                           *ln1_out,
                           linear1_weight,
                           &d_ln1_out,
                           d_linear1_weight);
    // tensor model parallel
    phi::fusion::AllReduce<T>(d_ln1_out, ring_id, dev_ctx);
    pre_layernorm_helper.LayerNormGrad(dev_ctx,
                                       d_ln1_out.data<T>(),
                                       x.data<T>(),
                                       ln1_gamma_ptr,
                                       ln1_mean->data<U>(),
                                       ln1_variance->data<U>(),
                                       d_x->data<T>(),
                                       d_ln1_gamma_ptr,
                                       d_ln1_beta_ptr);
  } else {
    MatMulGrad<T, Context>(
        dev_ctx, d_linear1_out, x, linear1_weight, d_x, d_linear1_weight);
    // tensor model parallel
    phi::fusion::AllReduce<T>(*d_x, ring_id, dev_ctx);
  }

  if (add_residual) {
    // gradient accumulation
    std::vector<const phi::DenseTensor*> ins = {&d_residual, d_x};
    std::vector<phi::DenseTensor*> outs = {d_x};
    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, ins, &outs, phi::funcs::AddFunctor<T>());
  }
}

template <typename T, typename Context>
void FusedFeedForwardGradKernel(
    const Context& dev_ctx,
    const DenseTensor& out_grad,
    const DenseTensor& x,
    const DenseTensor& linear1_weight,
    const DenseTensor& linear1_bias,
    const DenseTensor& linear2_weight,
    const DenseTensor& dropout1_mask,
    const DenseTensor& dropout2_mask,
    const DenseTensor& linear1_out,
    const DenseTensor& dropout1_out,
    const DenseTensor& dropout2_out,
    const paddle::optional<DenseTensor>& ln1_scale,
    const paddle::optional<DenseTensor>& ln1_bias,
    const paddle::optional<DenseTensor>& ln1_out,
    const paddle::optional<DenseTensor>& ln1_mean,
    const paddle::optional<DenseTensor>& ln1_variance,
    const paddle::optional<DenseTensor>& ln2_scale,
    const paddle::optional<DenseTensor>& ln2_bias,
    const paddle::optional<DenseTensor>& ln2_mean,
    const paddle::optional<DenseTensor>& ln2_variance,
    const paddle::optional<DenseTensor>& linear2_bias,
    bool pre_layer_norm,
    float ln1_epsilon,
    float ln2_epsilon,
    const std::string& act_method,
    float dropout1_prob,
    float dropout2_prob,
    const std::string& dropout1_implementation,
    const std::string& dropout2_implementation,
    bool is_test,
    bool dropout1_fix_seed,
    bool dropout2_fix_seed,
    int dropout1_seed_val,
    int dropout2_seed_val,
    bool add_residual,
    int ring_id,
    DenseTensor* x_grad,
    DenseTensor* ln1_scale_grad,
    DenseTensor* ln1_bias_grad,
    DenseTensor* ln2_scale_grad,
    DenseTensor* ln2_bias_grad,
    DenseTensor* linear1_weight_grad,
    DenseTensor* linear1_bias_grad,
    DenseTensor* linear2_weight_grad,
    DenseTensor* linear2_bias_grad) {
  using U = phi::funcs::LayerNormParamType<T>;

  auto d_out = &out_grad;
  auto x_ptr = x;

  auto dropout1_mask_ptr = &dropout1_mask;
  auto dropout2_mask_ptr = &dropout2_mask;
  auto linear1_out_ptr = &linear1_out;
  auto* ln1_out_ptr = pre_layer_norm ? ln1_out.get_ptr() : nullptr;
  auto* dropout2_out_ptr = &dropout2_out;
  auto linear1_weight_ptr = &linear1_weight;
  auto* linear1_bias_ptr = &linear1_bias;
  auto linear2_weight_ptr = &linear2_weight;
  auto* ln1_mean_ptr = pre_layer_norm ? ln1_mean.get_ptr() : nullptr;
  auto* ln1_variance_ptr = pre_layer_norm ? ln1_variance.get_ptr() : nullptr;
  auto* ln1_scale_ptr = pre_layer_norm ? ln1_scale.get_ptr() : nullptr;
  auto* ln1_bias_ptr = pre_layer_norm ? ln1_bias.get_ptr() : nullptr;
  auto* ln2_mean_ptr = !pre_layer_norm ? ln2_mean.get_ptr() : nullptr;
  auto* ln2_variance_ptr = !pre_layer_norm ? ln2_variance.get_ptr() : nullptr;
  auto* ln2_scale_ptr = !pre_layer_norm ? ln2_scale.get_ptr() : nullptr;
  auto* ln2_bias_ptr = !pre_layer_norm ? ln2_bias.get_ptr() : nullptr;

  auto* d_x = x_grad;
  auto* d_ln1_scale = pre_layer_norm ? ln1_scale_grad : nullptr;
  auto* d_ln1_bias = pre_layer_norm ? ln1_bias_grad : nullptr;
  auto* d_ln2_scale = pre_layer_norm ? nullptr : ln2_scale_grad;
  auto* d_ln2_bias = pre_layer_norm ? nullptr : ln2_bias_grad;
  auto* d_linear1_weight = linear1_weight_grad;
  auto* d_linear1_bias = linear1_bias_grad;
  auto* d_linear2_weight = linear2_weight_grad;
  auto* d_linear2_bias = linear2_bias_grad;

  bool is_upscale_in_train1 = dropout1_implementation == "upscale_in_train";
  bool is_upscale_in_train2 = dropout2_implementation == "upscale_in_train";

  phi::fusion::DropoutParam dropout_param1(dropout1_fix_seed,
                                           0,
                                           is_test,
                                           is_upscale_in_train1,
                                           dropout1_prob,
                                           nullptr,
                                           dropout1_seed_val);
  phi::fusion::DropoutParam dropout_param2(dropout2_fix_seed,
                                           0,
                                           is_test,
                                           is_upscale_in_train2,
                                           dropout2_prob,
                                           nullptr,
                                           dropout2_seed_val);

  dev_ctx.template Alloc<T>(d_x, d_x->numel() * sizeof(T));
  if (d_ln1_scale) {
    dev_ctx.template Alloc<U>(d_ln1_scale, d_ln1_scale->numel() * sizeof(U));
  }
  if (d_ln1_bias) {
    dev_ctx.template Alloc<U>(d_ln1_bias, d_ln1_bias->numel() * sizeof(U));
  }
  if (d_ln2_scale) {
    dev_ctx.template Alloc<U>(d_ln2_scale, d_ln2_scale->numel() * sizeof(U));
  }
  if (d_ln2_bias) {
    dev_ctx.template Alloc<U>(d_ln2_bias, d_ln2_bias->numel() * sizeof(U));
  }
  if (d_linear1_bias) {
    dev_ctx.template Alloc<T>(d_linear1_bias,
                              d_linear1_bias->numel() * sizeof(T));
  }
  if (d_linear2_bias) {
    dev_ctx.template Alloc<T>(d_linear2_bias,
                              d_linear2_bias->numel() * sizeof(T));
  }
  dev_ctx.template Alloc<T>(d_linear1_weight,
                            d_linear1_weight->numel() * sizeof(T));
  dev_ctx.template Alloc<T>(d_linear2_weight,
                            d_linear2_weight->numel() * sizeof(T));

  auto x_dim = x.dims();
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
      phi::RowMatrixFromVector(x_dim), 0, false);

  auto linear1_weight_dim = linear1_weight.dims();
  int d_model = linear1_weight_dim[0];
  int dim_feedforward = linear1_weight_dim[linear1_weight_dim.size() - 1];
  int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;

  FFNGrad<T, Context>(dev_ctx,
                      *d_out,
                      x,
                      dropout1_mask,
                      dropout2_mask,
                      linear1_out,
                      ln1_out_ptr,
                      dropout1_out,
                      dropout2_out_ptr,
                      linear1_weight,
                      linear1_bias_ptr,
                      linear2_weight,
                      ln1_scale_ptr,
                      ln1_bias_ptr,
                      ln1_mean_ptr,
                      ln1_variance_ptr,
                      ln2_scale_ptr,
                      ln2_bias_ptr,
                      ln2_mean_ptr,
                      ln2_variance_ptr,
                      d_x,
                      d_linear1_weight,
                      d_linear1_bias,
                      d_linear2_weight,
                      d_linear2_bias,
                      d_ln1_scale,
                      d_ln1_bias,
                      d_ln2_scale,
                      d_ln2_bias,
                      bsz_seq,
                      d_model,
                      dim_feedforward,
                      dropout_param1,
                      dropout_param2,
                      act_method,
                      pre_layer_norm,
                      ln1_epsilon,
                      ln2_epsilon,
                      add_residual,
                      ring_id);
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_feedforward,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedFeedForwardKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(fused_feedforward_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedFeedForwardGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
