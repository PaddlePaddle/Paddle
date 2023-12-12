// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/fusion/cutlass/variable_length_memory_efficient_attention.h"
#include "paddle/phi/kernels/fusion/gpu/block_attn.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/utils/none.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void BlockMultiheadAttentionKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv,
    const DenseTensor& key_cache,
    const DenseTensor& value_cache,
    const DenseTensor& seq_lens_encoder,
    const DenseTensor& seq_lens_decoder,
    const DenseTensor& seq_lens_this_time,
    const DenseTensor& padding_offsets,
    const DenseTensor& cum_offsets,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& block_tables,
    const paddle::optional<DenseTensor>& pre_key_cache,
    const paddle::optional<DenseTensor>& pre_value_cache,
    const paddle::optional<DenseTensor>& rope_emb,
    const paddle::optional<DenseTensor>& mask,
    const paddle::optional<DenseTensor>& tgt_mask,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    DenseTensor* fmha_out,
    DenseTensor* qkv_out,
    DenseTensor* key_cache_out,
    DenseTensor* value_cache_out) {
  dev_ctx.template Alloc<T>(fmha_out);
  InitValue(
      dev_ctx, fmha_out->data<T>(), fmha_out->numel(), static_cast<T>(0.));
  const auto& input_dims = qkv.dims();
  const auto& key_cache_dims = key_cache.dims();
  const int token_num = input_dims[0];
  const int num_head = key_cache_dims[1];
  const int dim_head = key_cache_dims[3];
  const int bsz = cum_offsets.dims()[0];
  const int max_block_per_seq = block_tables.dims()[1];
  VLOG(3) << "bsz: " << bsz << " token_num: " << token_num
          << " num_head: " << num_head << " dim_head: " << dim_head
          << " max_block_per_seq: " << max_block_per_seq;
  VLOG(3) << "fmha_out_dims: " << fmha_out->dims();

  bool causual = true;
  if (mask) {
    causual = false;
  }

  bool use_pre_cache = false;
  int pre_cache_length = 0;
  if (pre_key_cache) {
    use_pre_cache = true;
    pre_cache_length = pre_key_cache.get().dims()[2];
  }
  VLOG(3) << "token_num: " << token_num
          << " pre_cache_length: " << pre_cache_length;

  phi::DenseTensor max_dec_len_tensor;
  max_dec_len_tensor.Resize({{1}});
  auto* max_dec_len_data = dev_ctx.template Alloc<int>(
      &max_dec_len_tensor, max_dec_len_tensor.numel() * sizeof(int));
  int max_dec_len_this_time =
      GetMaxLen(dev_ctx, seq_lens_decoder, &max_dec_len_tensor, bsz);

  phi::DenseTensor max_enc_len_tensor;
  max_enc_len_tensor.Resize({{1}});
  auto* max_enc_len_data = dev_ctx.template Alloc<int>(
      &max_enc_len_tensor, max_enc_len_tensor.numel() * sizeof(int));
  int max_enc_len_this_time =
      GetMaxLen(dev_ctx, seq_lens_encoder, &max_enc_len_tensor, bsz);

  phi::DenseTensor qkv_out_decoder;
  if (max_dec_len_this_time > 0) {
    qkv_out_decoder.Resize({{bsz, 3, num_head, dim_head}});
    auto* qkv_out_decoder_data = dev_ctx.template Alloc<T>(
        &qkv_out_decoder, qkv_out_decoder.numel() * sizeof(T));
  }
  VLOG(3) << "max_len end";
  phi::DenseTensor unpadding_q, unpadding_k, unpadding_v;
  phi::DenseTensor softmax_out, softmax_lse, seed_offset;
  phi::DenseTensor q_trans, k_trans, v_trans, qktv_out;
  if (max_enc_len_this_time > 0) {
    if (!use_pre_cache) {
      unpadding_q.Resize({{token_num, num_head, dim_head}});
      unpadding_k.Resize({{token_num, num_head, dim_head}});
      unpadding_v.Resize({{token_num, num_head, dim_head}});

      dev_ctx.template Alloc<T>(&unpadding_q, unpadding_q.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&unpadding_k, unpadding_k.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&unpadding_v, unpadding_v.numel() * sizeof(T));
    } else {
      q_trans.Resize({{bsz, num_head, max_enc_len_this_time, dim_head}});
      k_trans.Resize({{bsz,
                       num_head,
                       max_enc_len_this_time + pre_cache_length,
                       dim_head}});
      v_trans.Resize({{bsz,
                       num_head,
                       max_enc_len_this_time + pre_cache_length,
                       dim_head}});
      qktv_out.Resize({{bsz, num_head, max_enc_len_this_time, dim_head}});

      dev_ctx.template Alloc<T>(&q_trans, q_trans.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&k_trans, k_trans.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&v_trans, v_trans.numel() * sizeof(T));
      dev_ctx.template Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
    }
  }
  VLOG(3) << "encoder";
  VLOG(3) << "max_enc_len_this_time: " << max_enc_len_this_time;
  if (max_enc_len_this_time > 0) {
    const int* sequence_lengths_data = seq_lens_encoder.data<int>();
    if (rope_emb) {
      rotary_qk_variable(dev_ctx,
                         qkv_out->data<T>(),
                         qkv.data<T>(),
                         rope_emb.get().data<float>(),
                         padding_offsets.data<int>(),
                         sequence_lengths_data,
                         token_num,
                         num_head,
                         max_seq_len,
                         rope_emb.get().dims()[2],
                         dim_head,
                         use_neox_style);
    }
    VLOG(3) << "rope end";
    VLOG(3) << "causual: " << causual;
    if (!use_pre_cache) {
      qkv_transpose_split<T>(dev_ctx,
                             unpadding_q.data<T>(),
                             unpadding_k.data<T>(),
                             unpadding_v.data<T>(),
                             qkv.data<T>(),
                             padding_offsets.data<int>(),
                             sequence_lengths_data,
                             token_num,
                             bsz,
                             num_head,
                             max_seq_len,
                             dim_head);
      VLOG(3) << "qkv split end";
      phi::FlashAttnUnpaddedKernel<T>(dev_ctx,
                                      unpadding_q,
                                      unpadding_k,
                                      unpadding_v,
                                      cu_seqlens_q,
                                      cu_seqlens_k,
                                      paddle::none /*fixed_seed_offset*/,
                                      causual ? paddle::none : mask,
                                      max_enc_len_this_time,
                                      max_enc_len_this_time,
                                      1.0f / sqrt(static_cast<float>(dim_head)),
                                      0.0,
                                      causual,
                                      false,
                                      true /* is_test*/,
                                      "" /*rng_name*/,
                                      fmha_out,
                                      &softmax_out,
                                      &softmax_lse,
                                      &seed_offset);
    } else {
      qkv_transpose_split<T>(
          dev_ctx,
          q_trans.data<T>(),
          k_trans.data<T>(),
          v_trans.data<T>(),
          qkv.data<T>(),
          pre_key_cache ? pre_key_cache.get().data<T>() : nullptr,
          pre_value_cache ? pre_value_cache.get().data<T>() : nullptr,
          padding_offsets.data<int>(),
          sequence_lengths_data,
          token_num,
          bsz,
          num_head,
          max_enc_len_this_time,
          max_seq_len,
          pre_cache_length,
          dim_head);
#ifdef PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION
      phi::fusion::MultiHeadAttentionVariableForwardKernel<T, phi::GPUContext>(
          dev_ctx,
          q_trans,
          k_trans,
          v_trans,
          seq_lens_encoder,
          seq_lens_encoder,
          mask,
          1.0f / sqrt(static_cast<float>(dim_head)),
          /*causual*/ false,
          pre_cache_length,
          &qktv_out);
#else
      PADDLE_THROW(phi::errors::Unimplemented(
          "Not supports MultiHeadAttentionVariableForwardKernel."));
#endif
      InvokeTransposeRemovePadding<T>(dev_ctx,
                                      qktv_out.data<T>(),
                                      sequence_lengths_data,
                                      fmha_out->data<T>(),
                                      bsz,
                                      num_head,
                                      max_enc_len_this_time,
                                      max_seq_len,
                                      dim_head,
                                      token_num,
                                      padding_offsets.data<int>());
    }
    VLOG(3) << "flash end";
    CacheKernel<T>(dev_ctx,
                   qkv,
                   block_tables,
                   padding_offsets,
                   seq_lens_encoder,
                   pre_key_cache,
                   pre_value_cache,
                   bsz,
                   token_num,
                   num_head,
                   dim_head,
                   max_seq_len,
                   pre_cache_length,
                   key_cache_out,
                   value_cache_out);
    VLOG(3) << "cache end";
  }
  VLOG(3) << "encoder done";
  VLOG(3) << "max_dec_len_this_time: " << max_dec_len_this_time;
  if (max_dec_len_this_time > 0) {
    GetDecoderTensor<T>(dev_ctx,
                        qkv,
                        nullptr,
                        cum_offsets.data<int>(),
                        &qkv_out_decoder,
                        nullptr,
                        token_num,
                        bsz,
                        num_head,
                        max_seq_len,
                        dim_head);
    VLOG(3) << "qkv_out_decoder: " << qkv_out_decoder.dims();
    blha<T>(dev_ctx,
            qkv_out_decoder,
            nullptr,  // qkv_bias
            &block_tables,
            tgt_mask ? &tgt_mask.get() : nullptr,
            &cum_offsets,
            &seq_lens_decoder,
            rope_emb ? &rope_emb.get() : nullptr,  // rope_emb
            key_cache_out,
            value_cache_out,
            fmha_out,
            bsz,
            max_block_per_seq,
            block_size,
            max_seq_len,
            pre_cache_length,
            num_head,
            dim_head,
            max_dec_len_this_time,
            rope_emb ? 1 : 0,
            1. / sqrt(dim_head),
            /*compute_bias*/ false,
            use_neox_style);
    VLOG(3) << "blha end";
  }
  VLOG(3) << "decoder done";
}

}  // namespace fusion
}  // namespace phi

#if CUDA_VERSION >= 11000 && defined(CUDA_BFLOAT16_AVALIABLE)
PD_REGISTER_KERNEL(block_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::BlockMultiheadAttentionKernel,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(block_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::BlockMultiheadAttentionKernel,
                   phi::dtype::float16) {}
#endif
