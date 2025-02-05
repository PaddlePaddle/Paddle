// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/device/gpu/gpu_resource_pool.h"
#include "paddle/phi/kernels/fusion/gpu/attention_layer.norm.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm_int8.h"
#include "paddle/phi/kernels/fusion/gpu/fmha_ref.h"
#include "paddle/phi/kernels/fusion/gpu/fused_attention_utils.h"
#include "paddle/phi/kernels/fusion/gpu/fused_multi_transformer_helper.cu.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void FusedMultiTransformerINT8OpKernel(
    const Context &dev_ctx,
    const DenseTensor &x_in,
    const std::vector<const DenseTensor *> &ln_scale,
    const std::vector<const DenseTensor *> &ln_bias,
    const std::vector<const DenseTensor *> &qkv_w,
    const paddle::optional<std::vector<const DenseTensor *>> &qkv_bias_in,
    const paddle::optional<std::vector<const DenseTensor *>> &cache_kv_in,
    const paddle::optional<DenseTensor> &time_step_in,
    const paddle::optional<DenseTensor> &src_mask_in,
    const std::vector<const DenseTensor *> &out_linear_w,
    const paddle::optional<std::vector<const DenseTensor *>> &out_linear_bias,
    const std::vector<const DenseTensor *> &ffn_ln_scale,
    const std::vector<const DenseTensor *> &ffn_ln_bias,
    const std::vector<const DenseTensor *> &ffn1_weight,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn1_bias,
    const std::vector<const DenseTensor *> &ffn2_weight,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn2_bias,
    const paddle::optional<std::vector<const DenseTensor *>> &qkv_out_scale,
    const paddle::optional<std::vector<const DenseTensor *>>
        &out_linear_out_scale,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn1_out_scale,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn2_out_scale,
    bool pre_layer_norm,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string &dropout_implementation,
    const std::string &act_method,
    bool trans_qkvw,
    int ring_id,
    int num_head_in,
    int dim_head_in,
    int dim_ffn_in,
    const std::vector<float> &qkv_in_scale,
    const std::vector<float> &out_linear_in_scale,
    const std::vector<float> &ffn1_in_scale,
    const std::vector<float> &ffn2_in_scale,
    int quant_round_type,
    float quant_max_bound,
    float quant_min_bound,
    std::vector<DenseTensor *> cache_kv_out,
    DenseTensor *out) {
  using U = phi::fusion::LayerNormParamType<T>;

  auto *time_step = time_step_in.get_ptr();

  // 0. input
  auto *input_x = &x_in;
  const auto input_x_dims = input_x->dims();
  int bsz = input_x_dims[0];
  int seq_len = input_x_dims[1];
  int dim_embed = input_x_dims[2];
  int bsz_seq = bsz * seq_len;

  // quant input scales, vector, size = num_layers

  // quant round type and bound

  // dequant output scales, tensor, size = [num_layers, n], n is gemm output
  // size
  auto qkv_out_scales = qkv_out_scale.get();
  auto out_linear_out_scales = out_linear_out_scale.get();
  auto ffn1_out_scales = ffn1_out_scale.get();
  auto ffn2_out_scales = ffn2_out_scale.get();

  // 1. layer norm
  auto ln_scales = ln_scale;
  auto ln_biases = ln_bias;

  auto ln_compute = phi::fusion::AttnLayerNorm<T, T, int8_t>(
      dev_ctx, epsilon, bsz_seq, dim_embed);
  phi::DenseTensor ln_mean, ln_var;
  ln_mean.Resize({{bsz_seq}});
  auto *ln_mean_data =
      dev_ctx.template Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
  ln_var.Resize({{bsz_seq}});
  auto *ln_var_data =
      dev_ctx.template Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

  // 2. qkv
  // x: qkv's input [batch_size, seq_len, dim_embed]
  // y: qkv's weight: [3, num_head, dim_head, dim_embed]
  auto qkv_weights = qkv_w;
  auto qkv_biases = qkv_bias_in.get();

  const auto qkv_w_dims = qkv_weights[0]->dims();
  int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
  int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
  int hidden_size = num_head * dim_head;
  int output_size = 3 * hidden_size;
  int input_size = dim_embed;

  bool compute_bias = qkv_biases.size() > 0 && time_step == nullptr;
  // (transA, transB, compute_bias) = (false, trans_qkvw, false)
  phi::fusion::AttnMatmulINT8<T> qkv_compute(
      dev_ctx, bsz_seq, output_size, input_size, compute_bias);
  phi::DenseTensor qkv_out;
  qkv_out.Resize({{bsz, seq_len, 3, num_head, dim_head}});
  auto *qkv_out_data =
      dev_ctx.template Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

  // 3. fmha
  phi::fusion::AttnDropoutParam attn_param(
      true, "upscale_in_train", 0.0, true, true, 0, nullptr);
  auto fmha_compute = phi::fusion::FMHARef<T>(
      dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
  auto *src_mask = src_mask_in.get_ptr();
  auto cache_kvs = std::vector<const DenseTensor *>();
  if (cache_kv_in) {
    cache_kvs = cache_kv_in.get();
  }
  auto cache_kv_outs = cache_kv_out;

  auto out_seq_len = seq_len;
  if (time_step) {
    PADDLE_ENFORCE_EQ(time_step->place(),
                      phi::CPUPlace(),
                      common::errors::PreconditionNotMet(
                          "The place of input(TimeStep) must be CPUPlace."));
    // cache_seq_len
    int time_step_value = time_step->data<int>()[0];
    PADDLE_ENFORCE_GT(
        time_step_value,
        0,
        common::errors::PreconditionNotMet(
            "The value of time_step must > 0, but now is %d", time_step_value));
    PADDLE_ENFORCE_EQ(
        seq_len,
        1,
        common::errors::PreconditionNotMet(
            "In decode stage, the seq_len of input must be 1, but now is %d",
            seq_len));
    out_seq_len += time_step_value;
  }

  phi::DenseTensor transpose_out_2, qk_out;
  transpose_out_2.Resize({{3, bsz, num_head, seq_len, dim_head}});
  auto *transpose_out_2_data = dev_ctx.template Alloc<T>(
      &transpose_out_2, transpose_out_2.numel() * sizeof(T));

  qk_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
  auto *qk_out_data =
      dev_ctx.template Alloc<T>(&qk_out, qk_out.numel() * sizeof(T));

  phi::DenseTensor softmax_out;
  phi::DenseTensor attn_dropout_mask_out, attn_dropout_out;
  phi::DenseTensor qktv_out, fmha_out;
  softmax_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
  auto *softmax_out_data =
      dev_ctx.template Alloc<T>(&softmax_out, softmax_out.numel() * sizeof(T));

  attn_dropout_mask_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
  auto *attn_dropout_mask_out_data = dev_ctx.template Alloc<T>(
      &attn_dropout_mask_out, attn_dropout_mask_out.numel() * sizeof(T));
  attn_dropout_out.Resize({{bsz, num_head, seq_len, out_seq_len}});
  auto *attn_dropout_data_data = dev_ctx.template Alloc<T>(
      &attn_dropout_out, attn_dropout_out.numel() * sizeof(T));

  qktv_out.Resize({{bsz, num_head, seq_len, dim_head}});
  auto *qktv_out_data =
      dev_ctx.template Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
  fmha_out.Resize({{bsz, seq_len, num_head, dim_head}});
  auto *fmha_out_data =
      dev_ctx.template Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

  // 4. out_linear
  auto out_linear_weights = out_linear_w;
  auto out_linear_biases = out_linear_bias.get();

  // (transA, transB, compute_bias) = (false, false, false)
  phi::fusion::AttnMatmulINT8<T> out_linear_compute(
      dev_ctx, bsz_seq, dim_embed, hidden_size, false);

  // 5. ln(residual + bias)
  phi::fusion::DropoutParam dropout_param2(
      true, 0, true, true, 0.0, nullptr, 0);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t, int32_t, int8_t>
      fused_dropout_layernorm_helper(
          dev_ctx, bsz_seq, dim_embed, dropout_param2, epsilon);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      fused_dropout_layernorm_helper_for_post_layernorm(
          dev_ctx, bsz_seq, dim_embed, dropout_param2, epsilon);
  auto ffn_ln_scales = ffn_ln_scale;
  auto ffn_ln_biases = ffn_ln_bias;
  phi::DenseTensor bias_dropout_residual_out, dropout_mask_out;
  T *bias_dropout_residual_out_data = nullptr;
  if (pre_layer_norm) {
    bias_dropout_residual_out.Resize({{bsz, seq_len, dim_embed}});
    bias_dropout_residual_out_data = dev_ctx.template Alloc<T>(
        &bias_dropout_residual_out,
        bias_dropout_residual_out.numel() * sizeof(T));
  }
  dropout_mask_out.Resize({{bsz, seq_len, dim_embed}});
  auto *dropout_mask_out_data = dev_ctx.template Alloc<uint8_t>(
      &dropout_mask_out, dropout_mask_out.numel() * sizeof(uint8_t));

  // 6. ffn matmul1
  auto ffn1_weights = ffn1_weight;
  auto ffn1_biases = ffn1_bias.get();
  auto ffn1_weight_dim = ffn1_weights[0]->dims();

  int dim_ffn = ffn1_weight_dim[0];
  phi::fusion::AttnMatmulINT8<T> ffn1_linear_compute(
      dev_ctx, bsz_seq, dim_ffn, dim_embed, false);
  phi::DenseTensor ffn1_out;
  ffn1_out.Resize({{bsz_seq, dim_ffn}});
  auto *ffn1_out_data =
      dev_ctx.template Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

  // 7. ffn act + bias
  phi::fusion::DropoutParam ffn1_dropout_param(
      true, 0, true, true, 0.0, nullptr, 0);
  phi::fusion::FusedDropoutHelper<T, uint8_t, int32_t, int8_t>
      fused_act_dropout_helper(dev_ctx, bsz_seq, dim_ffn, ffn1_dropout_param);
  phi::fusion::FusedDropoutHelper<T, uint8_t>
      fused_act_dropout_helper_for_post_layernorm(
          dev_ctx, bsz_seq, dim_ffn, ffn1_dropout_param);
  phi::DenseTensor ffn1_dropout_out, ffn1_dropout_mask;
  ffn1_dropout_out.Resize({{bsz_seq, dim_ffn}});
  auto *ffn1_dropout_out_data = dev_ctx.template Alloc<T>(
      &ffn1_dropout_out, ffn1_dropout_out.numel() * sizeof(T));
  ffn1_dropout_mask.Resize({{bsz_seq, dim_ffn}});
  auto *ffn1_dropout_mask_data = dev_ctx.template Alloc<uint8_t>(
      &ffn1_dropout_mask, ffn1_dropout_mask.numel() * sizeof(uint8_t));

  // 8. ffn2 matmul
  auto ffn2_weights = ffn2_weight;
  auto ffn2_biases = ffn2_bias.get();
  phi::fusion::AttnMatmulINT8<T> ffn2_linear_compute(
      dev_ctx, bsz_seq, dim_embed, dim_ffn, false);

  // 9. ffn2 residual bias
  phi::fusion::DropoutParam ffn2_dropout_param(
      true, 0, true, true, 0.0, nullptr, 0);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t, int32_t, int8_t>
      ffn2_fused_dropout_helper(
          dev_ctx, bsz_seq, dim_embed, ffn2_dropout_param, epsilon);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t, int32_t, T>
      ffn2_fused_dropout_dequant_helper(
          dev_ctx, bsz_seq, dim_embed, ffn2_dropout_param, epsilon);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      ffn2_fused_dropout_helper_for_post_layernorm(
          dev_ctx, bsz_seq, dim_embed, ffn2_dropout_param, epsilon);

  // []. init workspace for cublasLt transform
  phi::DenseTensor input_workspace, output_workspace, cublaslt_workspace;
  // for input and output transform data is CUBLASLT_ORDER_COL32 format,
  int m_max = bsz_seq, k_max = std::max(dim_embed, dim_ffn),
      n_max = std::max({output_size, dim_embed, dim_ffn});

  input_workspace.Resize({{(m_max * k_max + 31) / 32 * 32}});
  dev_ctx.template Alloc<int8_t>(&input_workspace,
                                 input_workspace.numel() * sizeof(int8_t));

  output_workspace.Resize({{(n_max * m_max + 31) / 32 * 32}});
  dev_ctx.template Alloc<int32_t>(&output_workspace,
                                  output_workspace.numel() * sizeof(int32_t));

  cublaslt_workspace.Resize({{3000000}});
  dev_ctx.template Alloc<int8_t>(&cublaslt_workspace,
                                 cublaslt_workspace.numel() * sizeof(int8_t));

  // calc
  auto *from_data = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));
  phi::DenseTensor *from_tensor = out;
  phi::DenseTensor tmp_out;
  tmp_out.Resize({{bsz, seq_len, dim_embed}});
  auto *tmp_out_data =
      dev_ctx.template Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

  auto *x_data = input_x->data<T>();
  phi::DenseTensor *buf0 = nullptr;
  phi::DenseTensor *buf1 = nullptr;

  // step0:  x   --> buf1
  // step1: buf1 --> buf0
  // step2: buf0 --> buf1
  int layers = qkv_weights.size();
  if (pre_layer_norm) {
    buf1 = out;
  } else {
    buf0 = &tmp_out;
    buf1 = out;
  }

  for (int i = 0; i < layers; ++i) {
    // step1. layer_norm
    if (i == 0 && pre_layer_norm) {
      auto *ln_scale_data = ln_scales[i]->data<U>();
      auto *ln_bias_data = ln_biases[i]->data<U>();
      // TODO(wangxi): can remove mean var in inference
      ln_compute.ComputeForward(x_data,
                                ln_scale_data,
                                ln_bias_data,
                                input_workspace.data<int8_t>(),
                                ln_mean_data,
                                ln_var_data,
                                nullptr,
                                0,
                                qkv_in_scale[i],
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound);
    }

    // step2. qkv
    const phi::DenseTensor *qkv_bias =
        qkv_biases.size() > 0 ? qkv_biases[i] : nullptr;
    // NOTE: in decoder stage, bias is fused in fmha
    const phi::DenseTensor *bias = time_step ? nullptr : qkv_bias;
    if (!pre_layer_norm && i == 0) {
      qkv_compute.ComputeForward(qkv_weights[i],
                                 input_x,
                                 &input_workspace,
                                 bias,
                                 &qkv_out,
                                 &output_workspace,
                                 &qkv_out,
                                 qkv_in_scale[i],
                                 qkv_out_scales[i],
                                 quant_round_type,
                                 quant_max_bound,
                                 quant_min_bound);
    } else if (!pre_layer_norm) {
      qkv_compute.ComputeForward(qkv_weights[i],
                                 buf1,
                                 &input_workspace,
                                 bias,
                                 &qkv_out,
                                 &output_workspace,
                                 &qkv_out,
                                 qkv_in_scale[i],
                                 qkv_out_scales[i],
                                 quant_round_type,
                                 quant_max_bound,
                                 quant_min_bound);
    } else {
      qkv_compute.ComputeForwardINT8ToT(qkv_weights[i],
                                        qkv_in_scale[i],
                                        &input_workspace,
                                        bias,
                                        &qkv_out,
                                        &output_workspace,
                                        &qkv_out,
                                        qkv_out_scales[i]);
    }

    // step3. fmha
    const phi::DenseTensor *cache_kv =
        cache_kvs.size() > 0 ? cache_kvs[i] : nullptr;
    phi::DenseTensor *cache_kv_out = cache_kv ? cache_kv_outs[i] : nullptr;

    int cache_bsz = 0;
    if (cache_kv) {
      cache_bsz = cache_kv->dims()[1];
    }

    if (time_step) {  // generation decoder stage
      // [2, batch_size, num_head, max_seq_len, head_size]
      int max_seq_len = cache_kv->dims()[3];
      phi::fusion::fmha<T>(dev_ctx,
                           qkv_out,
                           *qkv_bias,
                           src_mask,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           cache_kv_out,
                           &fmha_out,
                           bsz,
                           cache_bsz,
                           seq_len,
                           max_seq_len,
                           num_head,
                           dim_head,
                           time_step->data<int>()[0],
                           0,
                           1. / sqrt(dim_head));
    } else if (cache_kv_out) {  // generation context stage
      // TODO(wangxi): can remove dropout in inference
      fmha_compute.ComputeForward(qkv_out,
                                  nullptr,
                                  src_mask,
                                  &transpose_out_2,
                                  nullptr,
                                  &qk_out,
                                  nullptr,
                                  &softmax_out,
                                  &attn_dropout_mask_out,
                                  &attn_dropout_out,
                                  &qktv_out,
                                  &fmha_out);
      // [3, bsz, num_head, seq_len, head_dim]
      T *qkv_data = transpose_out_2_data;
      int64_t q_size = bsz * seq_len * num_head * dim_head;
      int64_t k_size = q_size;
      const T *q_ptr = qkv_data;
      const T *k_ptr = q_ptr + q_size;
      const T *v_ptr = k_ptr + k_size;

      // [2, bsz, num_head, max_seq_len, head_dim]
      int max_seq_len = cache_kv_out->dims()[3];
      T *cache_kv_data = cache_kv_out->data<T>();
      int64_t cache_k_size = bsz * num_head * max_seq_len * dim_head;

      T *cache_k_ptr = cache_kv_data;
      T *cache_v_ptr = cache_kv_data + cache_k_size;

      phi::fusion::write_cache_kv<T>(dev_ctx,
                                     cache_k_ptr,
                                     cache_v_ptr,
                                     k_ptr,
                                     v_ptr,
                                     bsz,
                                     num_head,
                                     seq_len,
                                     max_seq_len,
                                     dim_head);
    } else {  // not generation
      // TODO(wangxi): can remove dropout in inference
      fmha_compute.ComputeForward(qkv_out,
                                  cache_kv,
                                  src_mask,
                                  &transpose_out_2,
                                  cache_kv_out,
                                  &qk_out,
                                  nullptr,
                                  &softmax_out,
                                  &attn_dropout_mask_out,
                                  &attn_dropout_out,
                                  &qktv_out,
                                  &fmha_out);
    }

    if (pre_layer_norm) {
      out_linear_compute.ComputeForwardTToINT8(out_linear_weights[i],
                                               out_linear_in_scale[i],
                                               &fmha_out,
                                               &input_workspace,
                                               nullptr,
                                               &output_workspace,
                                               nullptr,
                                               quant_round_type,
                                               quant_max_bound,
                                               quant_min_bound);
      phi::fusion::AllReduce<int32_t>(output_workspace,
                                      ring_id,
                                      bsz * seq_len * num_head * dim_head,
                                      dev_ctx);
    } else {
      out_linear_compute.ComputeForward(out_linear_weights[i],
                                        &fmha_out,
                                        &input_workspace,
                                        nullptr,
                                        buf0,
                                        &output_workspace,
                                        nullptr,
                                        out_linear_in_scale[i],
                                        out_linear_out_scales[i],
                                        quant_round_type,
                                        quant_max_bound,
                                        quant_min_bound);
      phi::fusion::AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
    }

    // step5. ln(residual + dropout(input + bias))
    if (pre_layer_norm) {
      auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
      auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
      auto *out_linear_bias_data = out_linear_biases[i]->data<T>();

      // inplace
      // non-inplace: buf1 -> input_workspace
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          dev_ctx,
          output_workspace.data<int32_t>(),
          x_data,
          out_linear_bias_data,
          ln_scale_data,
          ln_bias_data,
          bias_dropout_residual_out_data,
          dropout_mask_out_data,
          input_workspace.data<int8_t>(),
          ln_mean_data,
          ln_var_data,
          out_linear_in_scale[i],
          out_linear_out_scales[i]->data<float>(),
          ffn1_in_scale[i],
          quant_round_type,
          quant_max_bound,
          quant_min_bound);
    } else {
      auto *ln_scale_data = ln_scales[i]->data<U>();
      auto *ln_bias_data = ln_biases[i]->data<U>();
      auto *out_linear_bias_data = out_linear_biases[i]->data<T>();
      auto *residual_data = (i == 0 ? x_data : buf1->data<T>());
      fused_dropout_layernorm_helper_for_post_layernorm
          .LayernormResidualDropoutBias(dev_ctx,
                                        buf0->data<T>(),
                                        residual_data,
                                        out_linear_bias_data,
                                        ln_scale_data,
                                        ln_bias_data,
                                        buf0->data<T>(),
                                        dropout_mask_out_data,
                                        buf1->data<T>(),
                                        ln_mean_data,
                                        ln_var_data);
    }

    // step6. ffn matmul1

    if (pre_layer_norm) {
      ffn1_linear_compute.ComputeForwardINT8ToINT8(
          ffn1_weights[i],
          &input_workspace,
          nullptr,
          &output_workspace,
          nullptr,
          cublaslt_workspace.data<int8_t>());
    } else {
      ffn1_linear_compute.ComputeForward(ffn1_weights[i],
                                         buf1,
                                         &input_workspace,
                                         nullptr,
                                         &ffn1_out,
                                         &output_workspace,
                                         nullptr,
                                         ffn1_in_scale[i],
                                         ffn1_out_scales[i],
                                         quant_round_type,
                                         quant_max_bound,
                                         quant_min_bound);
    }

    // step7. act bias
    // TODO(wangxi): remove dropout mask in inference
    if (pre_layer_norm) {
      fused_act_dropout_helper.DropoutActBias(dev_ctx,
                                              output_workspace.data<int32_t>(),
                                              ffn1_biases[i]->data<T>(),
                                              "gelu",
                                              input_workspace.data<int8_t>(),
                                              ffn1_dropout_mask_data,
                                              ffn1_in_scale[i],
                                              ffn1_out_scales[i]->data<float>(),
                                              ffn2_in_scale[i],
                                              quant_round_type,
                                              quant_max_bound,
                                              quant_min_bound);
    } else {
      fused_act_dropout_helper_for_post_layernorm.DropoutActBias(
          dev_ctx,
          ffn1_out_data,
          ffn1_biases[i]->data<T>(),
          "gelu",
          ffn1_dropout_out_data,
          ffn1_dropout_mask_data);
    }

    // step8. ffn matmul2
    if (pre_layer_norm) {
      ffn2_linear_compute.ComputeForwardINT8ToINT8(
          ffn2_weights[i],
          &input_workspace,
          nullptr,
          &output_workspace,
          nullptr,
          cublaslt_workspace.data<int8_t>());
    } else {
      ffn2_linear_compute.ComputeForward(ffn2_weights[i],
                                         &ffn1_dropout_out,
                                         &input_workspace,
                                         nullptr,
                                         buf0,
                                         &output_workspace,
                                         nullptr,
                                         ffn2_in_scale[i],
                                         ffn2_out_scales[i],
                                         quant_round_type,
                                         quant_max_bound,
                                         quant_min_bound);
    }

    if (pre_layer_norm) {
      phi::fusion::AllReduce<int32_t>(output_workspace,
                                      ring_id,
                                      bsz * seq_len * num_head * dim_head,
                                      dev_ctx);
    } else {
      phi::fusion::AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
    }

    // step9. residual bias
    if (pre_layer_norm) {
      // TODO(wangxi): remove dropout mask in inference
      if (i < layers - 1) {
        auto *ln_scale_data = ln_scales[i + 1]->data<U>();
        auto *ln_bias_data = ln_biases[i + 1]->data<U>();

        ffn2_fused_dropout_helper.LayernormResidualDropoutBias(
            dev_ctx,
            output_workspace.data<int32_t>(),
            bias_dropout_residual_out_data,
            ffn2_biases[i]->data<T>(),
            ln_scale_data,
            ln_bias_data,
            buf1->data<T>(),
            dropout_mask_out_data,
            input_workspace.data<int8_t>(),
            ln_mean_data,
            ln_var_data,
            ffn2_in_scale[i],
            ffn2_out_scales[i]->data<float>(),
            qkv_in_scale[i + 1],
            quant_round_type,
            quant_max_bound,
            quant_min_bound);
      } else {
        ffn2_fused_dropout_dequant_helper.ResidualDropoutBias(
            dev_ctx,
            output_workspace.data<int32_t>(),
            bias_dropout_residual_out_data,
            ffn2_biases[i]->data<T>(),
            buf1->data<T>(),
            dropout_mask_out_data,
            ffn2_in_scale[i],
            ffn2_out_scales[i]->data<float>(),
            1.0);
      }
    } else {
      auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
      auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
      ffn2_fused_dropout_helper_for_post_layernorm.LayernormResidualDropoutBias(
          dev_ctx,
          buf0->data<T>(),
          buf1->data<T>(),
          ffn2_biases[i]->data<T>(),
          ln_scale_data,
          ln_bias_data,
          buf0->data<T>(),
          dropout_mask_out_data,
          buf1->data<T>(),
          ln_mean_data,
          ln_var_data);
    }
    if (pre_layer_norm) {
      x_data = buf1->data<T>();
    }
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multi_transformer_int8,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMultiTransformerINT8OpKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
}
