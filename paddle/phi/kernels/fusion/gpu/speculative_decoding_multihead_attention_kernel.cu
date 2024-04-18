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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/fusion/gpu/block_attn.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/reshape_kernel.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void SpeculativeDecodingMultiheadAttentionKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv,        // [token_num, 3*hidden_dim]
    const DenseTensor& key_cache,  // [bsz, num_head, max_seq_len, dim_head]
    const DenseTensor& value_cache,
    const DenseTensor& seq_lens_encoder,
    const DenseTensor& seq_lens_decoder,
    const DenseTensor& seq_lens_this_time,
    const DenseTensor& padding_offsets,
    const DenseTensor& cum_offsets,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const paddle::optional<DenseTensor>& rope_emb,
    const paddle::optional<DenseTensor>& mask,
    const paddle::optional<DenseTensor>& qkv_bias,
    const int max_enc_len_this_time,
    const int max_dec_len_this_time,
    int token_num_in_cache,
    int max_seq_len,
    bool use_neox_style,
    const std::string& compute_dtype,
    DenseTensor* fmha_out,
    DenseTensor* qkv_out,
    DenseTensor* key_cache_out,
    DenseTensor* value_cache_out) {
  phi::DenseTensor qkv_buf;
  phi::DenseTensor fmha_buf;

  VLOG(1) << "fmha_out " << fmha_out->dims();
  dev_ctx.template Alloc<T>(fmha_out);
  fmha_buf = *fmha_out;

  InitValue(dev_ctx, fmha_buf.data<T>(), fmha_buf.numel(), static_cast<T>(0.));
  const auto& input_dims = qkv.dims();
  const auto& key_cache_dims = key_cache.dims();
  const int token_num = input_dims[0];
  const int num_head = key_cache_dims[2];
  const int dim_head = key_cache_dims[3];
  const int cache_token_num = key_cache_dims[0] * key_cache_dims[1];

  const int bsz = cum_offsets.dims()[0];
  VLOG(3) << "bsz: " << bsz << " token_num: " << token_num
          << " num_head: " << num_head << " dim_head: " << dim_head;
  VLOG(3) << "fmha_out_dims: " << fmha_out->dims();
  bool causual = true;
  if (mask) {
    causual = false;
  }

  VLOG(3) << "token_num: " << token_num;

  phi::DenseTensor max_dec_len_tensor;
  max_dec_len_tensor.Resize({{1}});
  auto* max_dec_len_data = dev_ctx.template Alloc<int>(
      &max_dec_len_tensor, max_dec_len_tensor.numel() * sizeof(int));

  phi::DenseTensor max_enc_len_tensor;
  max_enc_len_tensor.Resize({{1}});
  auto* max_enc_len_data = dev_ctx.template Alloc<int>(
      &max_enc_len_tensor, max_enc_len_tensor.numel() * sizeof(int));

  phi::DenseTensor unpadding_q, unpadding_k, unpadding_v,
      unpadding_k_after_cache, unpadding_v_after_cache;
  phi::DenseTensor softmax_out, softmax_lse, seed_offset;
  phi::DenseTensor q_trans, k_trans, v_trans, qktv_out;

  unpadding_q.Resize({{token_num, num_head, dim_head}});
  unpadding_k.Resize({{token_num, num_head, dim_head}});
  unpadding_v.Resize({{token_num, num_head, dim_head}});

  unpadding_k_after_cache.Resize(
      {{token_num_in_cache + token_num, num_head, dim_head}});
  unpadding_v_after_cache.Resize(
      {{token_num_in_cache + token_num, num_head, dim_head}});

  dev_ctx.template Alloc<T>(&unpadding_q, unpadding_q.numel() * sizeof(T));
  dev_ctx.template Alloc<T>(&unpadding_k, unpadding_k.numel() * sizeof(T));
  dev_ctx.template Alloc<T>(&unpadding_v, unpadding_v.numel() * sizeof(T));

  dev_ctx.template Alloc<T>(&unpadding_k_after_cache,
                            unpadding_k_after_cache.numel() * sizeof(T));
  dev_ctx.template Alloc<T>(&unpadding_v_after_cache,
                            unpadding_v_after_cache.numel() * sizeof(T));

  // qkv: [self.batch_size * self.seq_len, nd*hd]
  qkv_buf = qkv;
  *key_cache_out = key_cache;
  *value_cache_out = value_cache;

  if (max_enc_len_this_time > 0) {
    const int* sequence_lengths_data = seq_lens_encoder.data<int>();
    if (rope_emb) {
      rotary_qk_variable(dev_ctx,
                         qkv_buf.data<T>(),
                         qkv_buf.data<T>(),
                         rope_emb.get().data<float>(),
                         padding_offsets.data<int>(),
                         sequence_lengths_data,
                         token_num,
                         num_head,
                         max_seq_len,
                         rope_emb.get().dims()[2],
                         dim_head,
                         use_neox_style);
      VLOG(3) << "rope end";
      VLOG(3) << "causual: " << causual;
    }

    VLOG(3) << "you got here1 !";
    qkv_transpose_split<T>(dev_ctx,
                           unpadding_q.data<T>(),
                           unpadding_k.data<T>(),
                           unpadding_v.data<T>(),
                           qkv_buf.data<T>(),
                           padding_offsets.data<int>(),
                           sequence_lengths_data,
                           token_num,
                           bsz,
                           num_head,
                           max_seq_len,
                           dim_head);

    phi::FlashAttnUnpaddedKernel<T>(
        dev_ctx,
        unpadding_q,
        unpadding_k,
        unpadding_v,
        cu_seqlens_q,
        cu_seqlens_k,
        paddle::none /*fixed_seed_offset*/,
        causual ? paddle::none : mask,
        max_enc_len_this_time,  // max_enc_len_this_time
        max_enc_len_this_time,  // max_enc_len_this_time
        1.0f / sqrt(static_cast<float>(dim_head)),
        0.0,
        causual,
        false,
        true /* is_test*/,
        "" /*rng_name*/,
        &fmha_buf,
        &softmax_out,
        &softmax_lse,
        &seed_offset);
    VLOG(3) << "-------fmha_buf dims: " << fmha_buf.dims();
  }

  if (max_dec_len_this_time > 0) {
    const int seq_len = token_num / bsz;
    const int* sequence_lengths_data = seq_lens_decoder.data<int>();
    if (rope_emb) {
      rotary_qk_variable_specu(dev_ctx,
                               qkv_buf.data<T>(),
                               qkv_buf.data<T>(),
                               rope_emb.get().data<float>(),
                               padding_offsets.data<int>(),
                               sequence_lengths_data,
                               token_num,
                               num_head,
                               max_seq_len,
                               token_num_in_cache,
                               rope_emb.get().dims()[2],
                               dim_head,
                               use_neox_style);
      VLOG(3) << "rope end!";
    }

    qkv_transpose_split<T>(dev_ctx,
                           unpadding_q.data<T>(),
                           unpadding_k.data<T>(),
                           unpadding_v.data<T>(),
                           qkv_buf.data<T>(),
                           padding_offsets.data<int>(),
                           sequence_lengths_data,
                           token_num,
                           bsz,
                           num_head,
                           max_seq_len,
                           dim_head);

    WriteCacheToKVKernel<T>(dev_ctx,
                            unpadding_k,
                            unpadding_v,
                            key_cache,
                            value_cache,
                            unpadding_k_after_cache,
                            unpadding_v_after_cache,
                            token_num_in_cache,
                            bsz,
                            token_num,
                            num_head,
                            dim_head);

    auto unpadding_q_reshaped = phi::Reshape<T, phi::GPUContext>(
        dev_ctx, unpadding_q, {bsz * seq_len, num_head, dim_head});
    auto unpadding_k_after_cache_reshaped = phi::Reshape<T, phi::GPUContext>(
        dev_ctx,
        unpadding_k_after_cache,
        {bsz * token_num_in_cache + seq_len, num_head, dim_head});
    auto unpadding_v_after_cache_reshaped = phi::Reshape<T, phi::GPUContext>(
        dev_ctx,
        unpadding_v_after_cache,
        {bsz * token_num_in_cache + seq_len, num_head, dim_head});

    phi::FlashAttnUnpaddedKernel<T>(
        dev_ctx,
        unpadding_q_reshaped,
        unpadding_k_after_cache_reshaped,
        unpadding_v_after_cache_reshaped,
        cu_seqlens_q,
        cu_seqlens_k,
        paddle::none /*fixed_seed_offset*/,
        causual ? paddle::none : mask,
        max_dec_len_this_time,  // max_dec_len_this_time
        max_dec_len_this_time,  // max_dec_len_this_time
        1.0f / sqrt(static_cast<float>(dim_head)),
        0.0,
        causual,
        false,
        true /* is_test*/,
        "" /*rng_name*/,
        &fmha_buf,
        &softmax_out,
        &softmax_lse,
        &seed_offset);
    VLOG(3) << "-------fmha_buf dims: " << fmha_buf.dims();
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(speculative_decoding_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::SpeculativeDecodingMultiheadAttentionKernel,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
