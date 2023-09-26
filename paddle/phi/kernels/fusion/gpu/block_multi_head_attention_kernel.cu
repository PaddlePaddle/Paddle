// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/fusion/gpu/block_multi_head_attention_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
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
    const paddle::optional<DenseTensor>& rope_emb,
    const paddle::optional<DenseTensor>& mask,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    DenseTensor* fmha_out,
    DenseTensor* qkv_out,
    DenseTensor* key_cache_out,
    DenseTensor* value_cache_out) {
  dev_ctx.template Alloc<T>(fmha_out);
  const auto& input_dims = qkv.dims();
  const auto& key_cache_dims = key_cache.dims();
  const int token_num = input_dims[0];
  const int num_head = key_cache_dims[1];
  const int dim_head = key_cache_dims[3];
  const int bsz = cum_offsets.dims()[0];
  const int max_block_per_seq = block_tables.dims()[1];
  VLOG(1) << "bsz: " << bsz << " token_num: " << token_num
          << " num_head: " << num_head << " dim_head: " << dim_head
          << " max_block_per_seq: " << max_block_per_seq;
  VLOG(1) << "fmha_out_dims: " << fmha_out->dims();

  bool causual = true;
  if (mask) {
    causual = false;
  }

  phi::DenseTensor max_len_tensor;

  max_len_tensor.Resize({{1}});
  auto* max_len_data = dev_ctx.template Alloc<int>(
      &max_len_tensor, max_len_tensor.numel() * sizeof(int));
  int max_len_this_time =
      GetMaxLen(dev_ctx, seq_lens_this_time, &max_len_tensor, bsz);

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
  VLOG(1) << "max_len end";
  phi::DenseTensor unpadding_q, unpadding_k, unpadding_v;
  phi::DenseTensor softmax_out, softmax_lse, seed_offset;
  if (max_enc_len_this_time > 0) {
    unpadding_q.Resize({{token_num, num_head, dim_head}});
    unpadding_k.Resize({{token_num, num_head, dim_head}});
    unpadding_v.Resize({{token_num, num_head, dim_head}});

    dev_ctx.template Alloc<T>(&unpadding_q, unpadding_q.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&unpadding_k, unpadding_k.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&unpadding_v, unpadding_v.numel() * sizeof(T));
  }
  VLOG(1) << "encoder";
  VLOG(1) << "max_enc_len_this_time: " << max_enc_len_this_time;
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
    VLOG(1) << "rope end";
    VLOG(1) << "causual: " << causual;
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
    VLOG(1) << "qkv split end";
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
    VLOG(1) << "flash end";
    CacheKernel<T>(dev_ctx,
                   qkv,
                   block_tables,
                   padding_offsets,
                   seq_lens_encoder,
                   token_num,
                   num_head,
                   dim_head,
                   max_seq_len,
                   key_cache_out,
                   value_cache_out);
    VLOG(1) << "cache end";
  }
  VLOG(1) << "encoder done";
  VLOG(1) << "max_dec_len_this_time: " << max_dec_len_this_time;
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
    VLOG(1) << "qkv_out_decoder: " << qkv_out_decoder.dims();
    blha<T>(dev_ctx,
            qkv_out_decoder,
            nullptr,  // qkv_bias
            &block_tables,
            nullptr,  // not need mask during generation
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
            num_head,
            dim_head,
            max_dec_len_this_time,
            rope_emb ? 1 : 0,
            1. / sqrt(dim_head),
            /*compute_bias*/ false,
            use_neox_style);
    VLOG(1) << "blha end";
  }
  VLOG(1) << "decoder done";
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(block_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::BlockMultiheadAttentionKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
