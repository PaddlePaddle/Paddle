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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/fusion/gpu/fmha_ref.h"
#include "paddle/phi/kernels/fusion/gpu/fused_attention_utils.h"
#include "paddle/phi/kernels/fusion/gpu/fused_multi_transformer_helper.cu.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedMultiTransformerOpKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const std::vector<const DenseTensor *> &ln_scales,
    const paddle::optional<std::vector<const DenseTensor *>> &ln_biases_in,
    const std::vector<const DenseTensor *> &qkv_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &qkv_biases_in,
    const paddle::optional<std::vector<const DenseTensor *>> &cache_kvs_in,
    const paddle::optional<std::vector<const DenseTensor *>> &pre_caches_in,
    const paddle::optional<DenseTensor> &rotary_tensor_in,
    const paddle::optional<DenseTensor> &beam_offset,
    const paddle::optional<DenseTensor> &time_step_in,
    const paddle::optional<DenseTensor> &seq_lengths,
    const paddle::optional<DenseTensor> &src_mask_in,
    const std::vector<const DenseTensor *> &out_linear_weights,
    const paddle::optional<std::vector<const DenseTensor *>>
        &out_linear_biases_in,
    const std::vector<const DenseTensor *> &ffn_ln_scales,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn_ln_biases_in,
    const std::vector<const DenseTensor *> &ffn1_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn1_biases_in,
    const std::vector<const DenseTensor *> &ffn2_weights,
    const paddle::optional<std::vector<const DenseTensor *>> &ffn2_biases_in,
    bool pre_layer_norm,
    float epsilon,
    float residual_alpha,
    float dropout_rate,
    int rotary_emb_dims,
    bool is_test,
    const std::string &dropout_implementation,
    const std::string &act_method,
    bool trans_qkvw,
    int ring_id,
    const std::string &norm_type,
    bool use_neox_rotary_style,
    int gqa_group_size,
    std::vector<DenseTensor *> cache_kv_outs,
    DenseTensor *out) {
  using U = phi::fusion::LayerNormParamType<T>;

  auto *time_step = time_step_in.get_ptr();
  // 0. input
  auto *input_x = &x;
  const auto input_x_dims = input_x->dims();
  int bsz = input_x_dims[0];
  int seq_len = input_x_dims[1];
  int dim_embed = input_x_dims[2];
  int bsz_seq = bsz * seq_len;

  // Optional Bias input for LayerNorm / RMSNorm
  std::vector<const DenseTensor *> ln_biases;
  auto ln_biases_in_ptr = ln_biases_in.get_ptr();
  if (ln_biases_in_ptr) {
    ln_biases = *ln_biases_in_ptr;
  }

  std::vector<const DenseTensor *> ffn_ln_biases;
  auto ffn_ln_biases_in_ptr = ffn_ln_biases_in.get_ptr();
  if (ffn_ln_biases_in_ptr) {
    ffn_ln_biases = *ffn_ln_biases_in_ptr;
  }

  bool use_glu = (act_method == "geglu" || act_method == "swiglu");

  bool remove_padding = false;
  auto *sequence_lengths = seq_lengths.get_ptr();
  phi::DenseTensor sequence_lengths_backup;
  if (sequence_lengths) {
    remove_padding = true;
  } else {
    sequence_lengths_backup.Resize({{1}});
    auto *sequence_lengths_backup_data = dev_ctx.template Alloc<int>(
        &sequence_lengths_backup,
        sequence_lengths_backup.numel() * sizeof(int));
    phi::fusion::InitValue(dev_ctx,
                           sequence_lengths_backup_data,
                           sequence_lengths_backup.numel() * sizeof(int),
                           static_cast<int>(seq_len));
    remove_padding = true;
  }

  auto *beam_cache_offset = beam_offset.get_ptr();
  int beam_size = 1;
  if (beam_cache_offset) {
    beam_size = beam_cache_offset->dims()[1];
  }

  phi::DenseTensor d_token_tensor;
  phi::DenseTensor padding_offset_tensor;
  phi::DenseTensor x_remove_padding;

  // cumulative seqlens [batch_size+1]
  phi::DenseTensor cu_seqlens_q, cu_seqlens_k;
  bool encoder_remove_padding = (remove_padding && !time_step);
  int token_num = 0;

  auto *from_data = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

  // Init out
  if (encoder_remove_padding) {
    phi::fusion::InitValue(
        dev_ctx, from_data, out->numel(), static_cast<T>(0.));
    phi::fusion::InitValue(
        dev_ctx, from_data, out->numel(), static_cast<T>(0.));
  }

  // remove padding in encoder
  if (encoder_remove_padding) {
    // just for encoder
    d_token_tensor.Resize({{1}});
    auto *d_token_num = dev_ctx.template Alloc<int>(
        &d_token_tensor, d_token_tensor.numel() * sizeof(int));
    // alloc the max size of padding_offset_tensor
    padding_offset_tensor.Resize({{bsz_seq}});
    dev_ctx.template Alloc<int>(&padding_offset_tensor,
                                padding_offset_tensor.numel() * sizeof(int));
    cu_seqlens_q.Resize({{bsz + 1}});
    dev_ctx.template Alloc<int32_t>(&cu_seqlens_q,
                                    cu_seqlens_q.numel() * sizeof(int32_t));

    phi::fusion::InvokeGetPaddingOffset(
        dev_ctx,
        &token_num,
        d_token_num,
        padding_offset_tensor.data<int>(),
        cu_seqlens_q.data<int>(),
        sequence_lengths ? sequence_lengths->data<int>()
                         : sequence_lengths_backup.data<int>(),
        bsz,
        seq_len);
    if (token_num == 0) return;
    padding_offset_tensor.Resize({{token_num}});
    x_remove_padding.Resize({{token_num, dim_embed}});
    dev_ctx.template Alloc<T>(&x_remove_padding,
                              x_remove_padding.numel() * sizeof(T));
    phi::fusion::InvokeRemovePadding(dev_ctx,
                                     x_remove_padding.data<T>(),
                                     input_x->data<T>(),
                                     padding_offset_tensor.data<int>(),
                                     token_num,
                                     dim_embed);
  } else {
    token_num = bsz_seq;
    if (token_num == 0) return;
  }

  auto *padding_offset_data =
      encoder_remove_padding ? padding_offset_tensor.data<int>() : nullptr;

  // 1. layer norm
  phi::fusion::NormHelper<T> norm_helper(
      dev_ctx, norm_type, token_num, dim_embed, epsilon, residual_alpha);
  phi::DenseTensor ln_mean, ln_var;
  ln_mean.Resize({{token_num}});
  auto *ln_mean_data =
      dev_ctx.template Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
  ln_var.Resize({{token_num}});
  auto *ln_var_data =
      dev_ctx.template Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

  // 2. qkv
  // x: qkv's input [batch_size, seq_len, dim_embed]
  // y: qkv's weight: [3, num_head, dim_head, dim_embed] if not GQA else
  // [num_head + 2 * gqa_group_size, dim_head, dim_embed]
  std::vector<const DenseTensor *> qkv_biases;
  auto qkv_biases_in_ptr = qkv_biases_in.get_ptr();
  if (qkv_biases_in_ptr) {
    qkv_biases = *qkv_biases_in_ptr;
  }

  const auto qkv_w_dims = qkv_weights[0]->dims();
  int num_head, dim_head;
  if (gqa_group_size > 0) {
    num_head = trans_qkvw ? (qkv_w_dims[0] - 2 * gqa_group_size)
                          : (qkv_w_dims[1] - 2 * gqa_group_size);
    dim_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
  } else {
    num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
    dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
  }
  int hidden_size = num_head * dim_head;
  int output_size = gqa_group_size <= 0
                        ? 3 * hidden_size
                        : (num_head + 2 * gqa_group_size) * dim_head;
  int input_size = dim_embed;

  // Set a flag whether need to add Matmul / Layernorm bias.
  bool compute_bias = qkv_biases.size() > 0;
  bool compute_ln_bias = ln_biases.size() > 0;

  auto qkv_compute = phi::fusion::GEMMHelper<T>(
      dev_ctx, token_num, output_size, input_size, "None", trans_qkvw);

  phi::DenseTensor qkv_out;
  if (gqa_group_size > 0) {
    qkv_out.Resize({{token_num, num_head + 2 * gqa_group_size, dim_head}});
  } else {
    qkv_out.Resize({{token_num, 3, num_head, dim_head}});
  }
  auto *qkv_out_data =
      dev_ctx.template Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

  // 2.1 rotary
  auto *rotary_tensor = rotary_tensor_in.get_ptr();

  // 3. fmha
  phi::fusion::AttnDropoutParam attn_param(
      true, "upscale_in_train", 0.0, true, true, 0, nullptr);

  auto *src_mask = src_mask_in.get_ptr();

  std::vector<const DenseTensor *> cache_kvs;
  auto cache_kvs_in_ptr = cache_kvs_in.get_ptr();
  if (cache_kvs_in_ptr) {
    cache_kvs = *cache_kvs_in_ptr;
  }

  std::vector<const DenseTensor *> pre_caches;
  auto pre_caches_ptr = pre_caches_in.get_ptr();
  if (pre_caches_ptr) {
    pre_caches = *pre_caches_ptr;
  }

  int cache_offset = 0;
  if (pre_caches.size() > 0) {
    cache_offset = pre_caches[0]->dims()[3];
  }

  auto out_seq_len = seq_len;
  if (time_step) {
    PADDLE_ENFORCE_EQ(
        time_step->place().GetType(),
        phi::AllocationType::CPU,
        common::errors::InvalidArgument(
            "The place of input(time_step) must be CPU, but got %s.",
            time_step->place()));
    // cache_seq_len
    int time_step_value = time_step->data<int>()[0];
    PADDLE_ENFORCE_GT(
        time_step_value,
        0,
        common::errors::InvalidArgument(
            "The value of time_step must > 0, but now is %d", time_step_value));
    PADDLE_ENFORCE_EQ(
        seq_len,
        1,
        common::errors::PreconditionNotMet(
            "In decode stage, the seq_len of input must be 1, but now is %d",
            seq_len));
    out_seq_len += time_step_value;
  } else {
    out_seq_len += cache_offset;
  }

  // whether to broadcast 2nd dimension for src_mask, default true
  // if mask_broadcast_num_heads if False, which means src_mask shape
  // will be:
  // 1. [batch_size, num_head, seq_len, seq_len] for encoder
  // 2. [batch_size, num_heads, 1, time_step+1] for decoder
  // and do not need to broadcast num_heads dimension when calculating
  // attn_mask offset in MHA
  bool mask_broadcast_num_heads = true;
  if (src_mask) {
    if (src_mask->dims()[1] == 1) {
      mask_broadcast_num_heads = true;
    } else if (src_mask->dims()[1] == num_head) {
      mask_broadcast_num_heads = false;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unknown dimension for attn_mask, the num_head(2nd) "
          "dimension is invalid, it should be 1 or num_head(%d), "
          "but got %d",
          num_head,
          src_mask->dims()[1]));
    }
  }

  phi::DenseTensor q_transpose_out, kv_transpose_out;
  q_transpose_out.Resize({{bsz, num_head, seq_len, dim_head}});
  auto *q_transpose_out_data = dev_ctx.template Alloc<T>(
      &q_transpose_out, q_transpose_out.numel() * sizeof(T));

  kv_transpose_out.Resize({{2, bsz, num_head, seq_len, dim_head}});
  auto *kv_transpose_out_data = dev_ctx.template Alloc<T>(
      &kv_transpose_out, kv_transpose_out.numel() * sizeof(T));

  if (encoder_remove_padding) {
    phi::fusion::InitValue(dev_ctx,
                           q_transpose_out_data,
                           q_transpose_out.numel(),
                           static_cast<T>(0.));
    phi::fusion::InitValue(dev_ctx,
                           kv_transpose_out_data,
                           kv_transpose_out.numel(),
                           static_cast<T>(0.));
  }

  // [2, bs, num_head, cache_seq_len + seq_len, head_dim]
  phi::DenseTensor pre_cache_kv_out;
  if (cache_offset > 0) {
    pre_cache_kv_out.Resize(
        {{2, bsz, num_head, seq_len + cache_offset, dim_head}});
    auto *pre_cache_kv_out_data = dev_ctx.template Alloc<T>(
        &pre_cache_kv_out, pre_cache_kv_out.numel() * sizeof(T));
  }

  phi::DenseTensor softmax_out;
  phi::DenseTensor qktv_out, fmha_out;

  // unpadding_q/unpadding_k/unpadding_v: [token_num, num_head, dim_head]
  phi::DenseTensor unpadding_q, unpadding_k, unpadding_v;
  phi::DenseTensor softmax_lse, seed_offset;

  unpadding_q.Resize({{token_num, num_head, dim_head}});
  if (gqa_group_size > 0) {
    unpadding_k.Resize({{token_num, gqa_group_size, dim_head}});
    unpadding_v.Resize({{token_num, gqa_group_size, dim_head}});
  } else {
    unpadding_k.Resize({{token_num, num_head, dim_head}});
    unpadding_v.Resize({{token_num, num_head, dim_head}});
  }
  cu_seqlens_k.Resize(cu_seqlens_q.dims());

  dev_ctx.template Alloc<T>(&unpadding_q, unpadding_q.numel() * sizeof(T));
  dev_ctx.template Alloc<T>(&unpadding_k, unpadding_k.numel() * sizeof(T));
  dev_ctx.template Alloc<T>(&unpadding_v, unpadding_v.numel() * sizeof(T));
  dev_ctx.template Alloc<int32_t>(&cu_seqlens_k,
                                  cu_seqlens_k.numel() * sizeof(int32_t));

  T *attn_dropout_mask_out_data = nullptr;
  T *attn_dropout_data_data = nullptr;

  qktv_out.Resize({{bsz, num_head, seq_len, dim_head}});
  auto *qktv_out_data =
      dev_ctx.template Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
  if (remove_padding) {
    fmha_out.Resize({{token_num, num_head, dim_head}});
  } else {
    fmha_out.Resize({{bsz, seq_len, num_head, dim_head}});
  }
  auto *fmha_out_data =
      dev_ctx.template Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));

  // 4. out_linear
  std::vector<const DenseTensor *> out_linear_biases;
  auto out_linear_biases_in_ptr = out_linear_biases_in.get_ptr();
  if (out_linear_biases_in_ptr) {
    out_linear_biases = *out_linear_biases_in_ptr;
  }

  // (transA, transB, compute_bias) = (false, false, false)

  auto out_linear_compute = phi::fusion::GEMMHelper<T>(
      dev_ctx, token_num, dim_embed, hidden_size, "None", false);

  // 5. ln(residual + bias)
  phi::DenseTensor bias_dropout_residual_out, dropout_mask_out;
  T *bias_dropout_residual_out_data = nullptr;
  if (pre_layer_norm) {
    bias_dropout_residual_out.Resize({{token_num, dim_embed}});
    bias_dropout_residual_out_data = dev_ctx.template Alloc<T>(
        &bias_dropout_residual_out,
        bias_dropout_residual_out.numel() * sizeof(T));
  }
  uint8_t *dropout_mask_out_data = nullptr;

  // 6. ffn matmul1
  std::vector<const DenseTensor *> ffn1_biases;
  auto ffn1_biases_in_ptr = ffn1_biases_in.get_ptr();
  if (ffn1_biases_in_ptr) {
    ffn1_biases = *ffn1_biases_in_ptr;
  }

  auto ffn1_weight_dim = ffn1_weights[0]->dims();
  // if quant weight,
  // matmul weight is transposed
  int dim_ffn = ffn1_weight_dim[1];
  phi::fusion::FFNHelper<T> ffn1_helper(
      dev_ctx, act_method, token_num, dim_ffn, dim_embed, "None");

  phi::DenseTensor ffn1_out;
  ffn1_out.Resize({{token_num, dim_ffn}});
  auto *ffn1_out_data =
      dev_ctx.template Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

  // Note(Zhengzekang): It is no need when using FP16 matmul.
  phi::DenseTensor mixgemm_workspace;
  char *mixgemm_workspace_data = nullptr;

  // 7. ffn act + bias
  phi::fusion::DropoutParam ffn1_dropout_param(
      true, 0, true, true, 0.0, nullptr, 0);
  phi::fusion::FusedDropoutHelper<T, int8_t> fused_act_dropout_helper(
      dev_ctx, token_num, dim_ffn, ffn1_dropout_param);
  phi::DenseTensor ffn1_dropout_out, ffn1_dropout_mask;
  int tmp_dim_ffn = dim_ffn;
  if (use_glu) tmp_dim_ffn /= 2;
  int8_t *ffn1_dropout_mask_data = nullptr;
  ffn1_dropout_out.Resize({{token_num, tmp_dim_ffn}});
  auto *ffn1_dropout_out_data = dev_ctx.template Alloc<T>(
      &ffn1_dropout_out, ffn1_dropout_out.numel() * sizeof(T));

  // 8. ffn2 matmul
  std::vector<const DenseTensor *> ffn2_biases;
  auto ffn2_biases_in_ptr = ffn2_biases_in.get_ptr();
  if (ffn2_biases_in_ptr) {
    ffn2_biases = *ffn2_biases_in_ptr;
  }

  auto ffn2_linear_compute = phi::fusion::GEMMHelper<T>(
      dev_ctx, token_num, dim_embed, tmp_dim_ffn, "None", false);

  // 9. ffn2 residual bias
  phi::fusion::DropoutParam ffn2_dropout_param(
      true, 0, true, true, 0.0, nullptr, 0);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      ffn2_fused_dropout_helper(
          dev_ctx, token_num, dim_embed, ffn2_dropout_param, epsilon);

  phi::DenseTensor tmp_out, tmp_out_rm_padding;
  tmp_out.Resize({{token_num, dim_embed}});
  if (encoder_remove_padding) {
    tmp_out_rm_padding.Resize({{token_num, dim_embed}});
    auto *tmp_out_rm_padding_data = dev_ctx.template Alloc<T>(
        &tmp_out_rm_padding, tmp_out_rm_padding.numel() * sizeof(T));
  }
  auto *tmp_out_data =
      dev_ctx.template Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

  const T *x_data;
  if (encoder_remove_padding) {
    x_data = x_remove_padding.data<T>();
  } else {
    x_data = input_x->data<T>();
  }
  phi::DenseTensor *buf0 = nullptr;
  phi::DenseTensor *buf1 = nullptr;

  // step0:  x   --> buf1
  // step1: buf1 --> buf0
  // step2: buf0 --> buf1
  int layers = qkv_weights.size();
  if (encoder_remove_padding) {
    // In the case of variable lengths, the padding needs to be rebuilt
    // eventually. So buf0 and buf1 do not need to be changed according to the
    // pre_layer_norm and the number of layers.
    buf0 = &tmp_out;
    buf1 = &tmp_out_rm_padding;
  } else {
    if (pre_layer_norm) {
      if (layers & 1) {
        // odd, set buf1 as out
        buf0 = &tmp_out;
        buf1 = out;
      } else {
        // even, set buf0 as out
        buf0 = out;
        buf1 = &tmp_out;
      }
    } else {
      buf0 = &tmp_out;
      buf1 = out;
    }
  }

  int time_step_value = out_seq_len - 1;
  int multi_block_attention_min_partition_size =
      static_cast<int>(FLAGS_multi_block_attention_min_partition_size);
  int max_num_partitions =
      (time_step_value + multi_block_attention_min_partition_size - 1) /
      multi_block_attention_min_partition_size;

  phi::DenseTensor partial_max_logits_tensor;
  partial_max_logits_tensor.Resize({{bsz, num_head, max_num_partitions}});
  phi::DenseTensor partial_expsum_tensor;
  partial_expsum_tensor.Resize({{bsz, num_head, max_num_partitions}});
  phi::DenseTensor partial_out_tensor;
  partial_out_tensor.Resize({{bsz, num_head, max_num_partitions, dim_head}});

  dev_ctx.template Alloc<float>(
      &partial_max_logits_tensor,
      partial_max_logits_tensor.numel() * sizeof(float));
  dev_ctx.template Alloc<float>(&partial_expsum_tensor,
                                partial_expsum_tensor.numel() * sizeof(float));
  dev_ctx.template Alloc<T>(&partial_out_tensor,
                            partial_out_tensor.numel() * sizeof(T));

  for (int i = 0; i < layers; ++i) {
    // step1. layer_norm
    if (i == 0 && pre_layer_norm) {
      norm_helper.Norm(x_data,
                       ln_scales[i],
                       compute_ln_bias ? ln_biases[i] : nullptr, /*norm_bias*/
                       &ln_mean,                                 /*mean*/
                       &ln_var,                                  /*var*/
                       buf1);
    }

    // step2. qkv
    // NOTE: In decoder stage, bias is fused in fmha. In encoder stage, bias
    // is fused in QKVBiasAddTransposeSplit
    const phi::DenseTensor *qkv_bias =
        qkv_biases.size() > 0 ? qkv_biases[i] : nullptr;
    if (!pre_layer_norm && i == 0) {
      const phi::DenseTensor *tmp_input_x =
          (encoder_remove_padding) ? &x_remove_padding : input_x;
      qkv_compute.Compute(tmp_input_x,
                          qkv_weights[i],
                          /*weight_scale*/ nullptr,
                          /*bias*/ nullptr,
                          &mixgemm_workspace,
                          &qkv_out);
    } else {
      qkv_compute.Compute(buf1,
                          qkv_weights[i],
                          /*weight_scale*/ nullptr,
                          /*bias*/ nullptr,
                          &mixgemm_workspace,
                          &qkv_out);
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
      if (FLAGS_fused_multi_transformer_op_use_mbfmha) {
        int max_seq_len = cache_kv->dims()[3];
        phi::fusion::mbfmha<T>(dev_ctx,
                               qkv_out,
                               *qkv_bias,
                               src_mask,
                               nullptr,
                               sequence_lengths,
                               rotary_tensor,
                               cache_kv_out,
                               &fmha_out,
                               &partial_max_logits_tensor,
                               &partial_expsum_tensor,
                               &partial_out_tensor,
                               bsz,
                               cache_bsz,
                               seq_len,
                               max_seq_len,
                               num_head,
                               dim_head,
                               time_step_value,
                               rotary_emb_dims,
                               1. / sqrt(dim_head),
                               mask_broadcast_num_heads,
                               compute_bias,
                               use_neox_rotary_style,
                               gqa_group_size);

      } else {
        // [2, batch_size, num_head, max_seq_len, head_size]
        int max_seq_len = cache_kv->dims()[3];

        phi::fusion::fmha<T>(dev_ctx,
                             qkv_out,
                             *qkv_bias,
                             src_mask,
                             nullptr,
                             sequence_lengths,
                             rotary_tensor,
                             beam_cache_offset,
                             cache_kv_out,
                             &fmha_out,
                             bsz,
                             cache_bsz,
                             seq_len,
                             max_seq_len,
                             num_head,
                             dim_head,
                             time_step_value,
                             rotary_emb_dims,
                             1. / sqrt(dim_head),
                             mask_broadcast_num_heads,
                             compute_bias,
                             use_neox_rotary_style,
                             gqa_group_size);
      }
    } else if (cache_kv_out) {  // generation context stage
      if (!encoder_remove_padding) {
        PADDLE_THROW(common::errors::InvalidArgument(
            "encoder_remove_padding must be True, but got False"));
      }
      if (rotary_emb_dims != 0) {
        if (gqa_group_size <= 0) {
          phi::fusion::rotary_qk_variable(
              dev_ctx,
              qkv_out_data,
              qkv_out_data,
              qkv_bias->data<T>(),
              rotary_tensor->data<float>(),
              padding_offset_data,
              sequence_lengths ? sequence_lengths->data<int>()
                               : sequence_lengths_backup.data<int>(),
              token_num,
              num_head,
              seq_len,
              rotary_tensor->dims()[3],
              dim_head,
              rotary_tensor->dims()[1]);
        } else {
          phi::fusion::gqa_rotary_qk_variable(
              dev_ctx,
              qkv_out_data,
              qkv_out_data,
              qkv_bias->data<T>(),
              rotary_tensor->data<float>(),
              padding_offset_data,
              sequence_lengths ? sequence_lengths->data<int>()
                               : sequence_lengths_backup.data<int>(),
              token_num,
              num_head,
              seq_len,
              rotary_tensor->dims()[3],
              dim_head,
              gqa_group_size,
              rotary_tensor->dims()[1]);
        }
      }
      if (gqa_group_size <= 0) {
        phi::fusion::qkv_transpose_split<T>(
            dev_ctx,
            unpadding_q.data<T>(),
            unpadding_k.data<T>(),
            unpadding_v.data<T>(),
            qkv_out_data,
            padding_offset_data,
            sequence_lengths ? sequence_lengths->data<int>()
                             : sequence_lengths_backup.data<int>(),
            token_num,
            bsz,
            num_head,
            seq_len,
            dim_head);
      } else {
        phi::fusion::gqa_qkv_transpose_split<T>(
            dev_ctx,
            unpadding_q.data<T>(),
            unpadding_k.data<T>(),
            unpadding_v.data<T>(),
            qkv_out_data,
            padding_offset_data,
            sequence_lengths ? sequence_lengths->data<int>()
                             : sequence_lengths_backup.data<int>(),
            token_num,
            bsz,
            num_head,
            seq_len,
            dim_head,
            gqa_group_size);
      }
      phi::Copy(
          dev_ctx, cu_seqlens_q, cu_seqlens_k.place(), false, &cu_seqlens_k);

      // fmha_out[token_num, num_head, dim_head]
      phi::FlashAttnUnpaddedKernel<T>(dev_ctx,
                                      unpadding_q,
                                      unpadding_k,
                                      unpadding_v,
                                      cu_seqlens_q,
                                      cu_seqlens_k,
                                      paddle::none /*fixed_seed_offset*/,
                                      paddle::none /*attn_mask*/,
                                      seq_len,
                                      seq_len,
                                      1.0f / sqrt(static_cast<float>(dim_head)),
                                      0.0,
                                      true /*causal*/,
                                      false,
                                      true /* is_test*/,
                                      "" /*rng_name*/,
                                      &fmha_out,
                                      &softmax_out,
                                      &softmax_lse,
                                      &seed_offset);
      // Note(@RichardWooSJTU): gqa_write_cachekv do not support pre_cache
      // and cache quantization
      phi::fusion::gqa_write_cachekv<T>(dev_ctx,
                                        cache_kv_out,
                                        unpadding_k,
                                        unpadding_v,
                                        padding_offset_tensor,
                                        *sequence_lengths,
                                        seq_len);
    } else {  // not generation
      // TODO(wangxi): can remove dropout in inference
      phi::fusion::qkv_bias_add_transpose_split<T>(
          dev_ctx,
          q_transpose_out_data,
          kv_transpose_out_data,
          qkv_out_data,
          qkv_bias ? qkv_bias->data<T>() : nullptr,
          padding_offset_data,
          token_num,
          bsz,
          num_head,
          seq_len,
          dim_head,
          compute_bias);

      // q_transpose_out_data [bs, head_num, seq_len, dim_head]
      // kv_transpose_out_data [2， bs, head_num, seq_len, dim_head]
      if (rotary_emb_dims != 0) {
        auto *rotary_emb_data = rotary_tensor->data<float>();
        const int *sequence_lengths_data =
            sequence_lengths ? sequence_lengths->data<int>()
                             : sequence_lengths_backup.data<int>();
        // encoder_remove_padding ? sequence_lengths->data<int>() : nullptr;
        phi::fusion::rotary_qk(dev_ctx,
                               q_transpose_out_data,
                               kv_transpose_out_data,
                               q_transpose_out_data,
                               kv_transpose_out_data,
                               rotary_emb_data,
                               sequence_lengths_data,
                               rotary_emb_dims,
                               rotary_tensor->dims()[1],
                               bsz,
                               num_head,
                               seq_len,
                               dim_head,
                               use_neox_rotary_style);
      }
      phi::DenseTensor *tmp_padding_offset_tensor =
          encoder_remove_padding ? &padding_offset_tensor : nullptr;

      if (encoder_remove_padding) {
        phi::fusion::TransposeSplit<T>(
            dev_ctx,
            unpadding_q.data<T>(),
            unpadding_k.data<T>(),
            unpadding_v.data<T>(),
            q_transpose_out.data<T>(),
            kv_transpose_out.data<T>(),
            padding_offset_data,
            sequence_lengths ? sequence_lengths->data<int>()
                             : sequence_lengths_backup.data<int>(),
            token_num,
            bsz,
            num_head,
            seq_len,
            dim_head);
        phi::Copy(
            dev_ctx, cu_seqlens_q, cu_seqlens_k.place(), false, &cu_seqlens_k);

        // fmha_out[token_num, num_head, dim_head]
        phi::FlashAttnUnpaddedKernel<T>(
            dev_ctx,
            unpadding_q,
            unpadding_k,
            unpadding_v,
            cu_seqlens_q,
            cu_seqlens_k,
            paddle::none /*fixed_seed_offset*/,
            paddle::none /*attn_mask*/,
            seq_len,
            seq_len,
            1.0f / sqrt(static_cast<float>(dim_head)),
            0.0,
            true /*causal*/,
            false,
            true /* is_test*/,
            "" /*rng_name*/,
            &fmha_out,
            &softmax_out,
            &softmax_lse,
            &seed_offset);
      }
    }
    if (pre_layer_norm) {
      out_linear_compute.Compute(&fmha_out,
                                 out_linear_weights[i],
                                 /*weight_scale*/ nullptr,
                                 /*bias*/ nullptr,
                                 &mixgemm_workspace,
                                 buf1);

      phi::fusion::AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
    } else {
      out_linear_compute.Compute(&fmha_out,
                                 out_linear_weights[i],
                                 /*weight_scale*/ nullptr,
                                 /*bias*/ nullptr,
                                 &mixgemm_workspace,
                                 buf0);

      phi::fusion::AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
    }

    // step5. ln(residual + dropout(input + bias))
    if (pre_layer_norm) {
      norm_helper.NormResidualBias(
          buf1->data<T>(),
          x_data,
          compute_bias ? out_linear_biases[i] : nullptr, /*skip_bias*/
          ffn_ln_scales[i],
          compute_ln_bias ? ffn_ln_biases[i] : nullptr, /*norm_bias*/
          &ln_mean,                                     /*mean*/
          &ln_var,                                      /*var*/
          &bias_dropout_residual_out,
          buf1);
    } else {
      auto *residual_data = (i == 0 ? x_data : buf1->data<T>());
      norm_helper.NormResidualBias(
          buf0->data<T>(),
          residual_data,
          compute_bias ? out_linear_biases[i] : nullptr, /*skip_bias*/
          ln_scales[i],
          compute_ln_bias ? ln_biases[i] : nullptr, /*norm_bias*/
          &ln_mean,                                 /*mean*/
          &ln_var,                                  /*var*/
          buf0,
          buf1);
    }
    // step6. ffn matmul1
    ffn1_helper.Compute(buf1,
                        ffn1_weights[i],
                        /*weight_scale*/ nullptr,
                        compute_bias ? ffn1_biases[i] : nullptr,
                        &mixgemm_workspace,
                        &ffn1_out,
                        &ffn1_dropout_out);

    // step7. ffn2 matmul
    if (pre_layer_norm) {
      ffn2_linear_compute.Compute(&ffn1_dropout_out,
                                  ffn2_weights[i],
                                  nullptr,
                                  /*bias*/ nullptr,
                                  &mixgemm_workspace,
                                  buf1);
    } else {
      ffn2_linear_compute.Compute(&ffn1_dropout_out,
                                  ffn2_weights[i],
                                  nullptr,
                                  /*bias*/ nullptr,
                                  &mixgemm_workspace,
                                  buf0);
    }

    if (pre_layer_norm) {
      phi::fusion::AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
    } else {
      phi::fusion::AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
    }

    // step8. residual bias
    // TODO(wangxi): remove dropout mask in inference
    if (pre_layer_norm) {
      // TODO(wangxi): remove dropout mask in inference
      if (i < layers - 1) {
        norm_helper.NormResidualBias(
            buf1->data<T>(),
            bias_dropout_residual_out_data,
            compute_bias ? ffn2_biases[i] : nullptr, /*skip_bias*/
            ln_scales[i + 1],
            compute_ln_bias ? ln_biases[i + 1] : nullptr, /*norm_bias*/
            &ln_mean,                                     /*mean*/
            &ln_var,                                      /*var*/
            buf1,
            buf0);
      } else {
        ffn2_fused_dropout_helper.ResidualDropoutBias(
            dev_ctx,
            buf1->data<T>(),
            bias_dropout_residual_out_data,
            compute_bias ? ffn2_biases[i]->data<T>() : nullptr,
            buf1->data<T>(),
            dropout_mask_out_data);
      }
    } else {
      norm_helper.NormResidualBias(
          buf0->data<T>(),
          buf1->data<T>(),
          compute_bias ? ffn2_biases[i] : nullptr, /*skip_bias*/
          ffn_ln_scales[i],
          compute_ln_bias ? ffn_ln_biases[i] : nullptr, /*norm_bias*/
          &ln_mean,                                     /*mean*/
          &ln_var,                                      /*var*/
          buf0,
          buf1);
    }

    if (pre_layer_norm) {
      x_data = buf1->data<T>();
      std::swap(buf0, buf1);
    }
  }

  if (encoder_remove_padding) {
    if (pre_layer_norm) {
      phi::fusion::InvokeRebuildPadding(dev_ctx,
                                        from_data,
                                        buf0->data<T>(),
                                        padding_offset_data,
                                        token_num,
                                        dim_embed);
    } else {
      phi::fusion::InvokeRebuildPadding(dev_ctx,
                                        from_data,
                                        buf1->data<T>(),
                                        padding_offset_data,
                                        token_num,
                                        dim_embed);
    }
  }
}

}  // namespace fusion
}  // namespace phi

#if CUDA_VERSION >= 11000
PD_REGISTER_KERNEL(fused_multi_transformer,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMultiTransformerOpKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(fused_multi_transformer,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMultiTransformerOpKernel,
                   phi::dtype::float16) {}
#endif
