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
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/fusion/gpu/block_attn.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/utils/none.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void SpeculativeDecodingMultiheadAttentionKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv, // [token_num, 3*hidden_dim]
    const DenseTensor& key_cache,  // [bsz, num_head, max_seq_len, dim_head]
    const DenseTensor& value_cache,
    const DenseTensor& seq_lens_encoder,
    const DenseTensor& seq_lens_decoder,
    const DenseTensor& seq_lens_this_time,
    const DenseTensor& padding_offsets,
    const DenseTensor& cum_offsets,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const paddle::optional<DenseTensor>& pre_key_cache,
    const paddle::optional<DenseTensor>& pre_value_cache,
    const paddle::optional<DenseTensor>& mask,
    const paddle::optional<DenseTensor>& qkv_bias,
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
  const int num_head = key_cache_dims[1];
  const int dim_head = key_cache_dims[3];
  const int cache_token_num = key_cache_dims[0] * key_cache_dims[2];

  const int bsz = cum_offsets.dims()[0];
  VLOG(3) << "bsz: " << bsz << " token_num: " << token_num
          << " num_head: " << num_head << " dim_head: " << dim_head;
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

  phi::DenseTensor unpadding_q, unpadding_k, unpadding_v, unpadding_k_after_cache, unpadding_v_after_cache;
  phi::DenseTensor softmax_out, softmax_lse, seed_offset;
  phi::DenseTensor q_trans, k_trans, v_trans, qktv_out;

  if (!use_pre_cache) {
    unpadding_q.Resize({{token_num, num_head, dim_head}});
    unpadding_k.Resize({{token_num, num_head, dim_head}});
    unpadding_v.Resize({{token_num, num_head, dim_head}});

    unpadding_k_after_cache.Resize({{token_num_in_cache + token_num, num_head, dim_head}});
    unpadding_v_after_cache.Resize({{token_num_in_cache + token_num, num_head, dim_head}});

    dev_ctx.template Alloc<T>(&unpadding_q, unpadding_q.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&unpadding_k, unpadding_k.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&unpadding_v, unpadding_v.numel() * sizeof(T));
    
    dev_ctx.template Alloc<T>(&unpadding_k_after_cache, unpadding_k_after_cache.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&unpadding_v_after_cache, unpadding_v_after_cache.numel() * sizeof(T));

  } else {
    q_trans.Resize({{bsz, num_head, max_enc_len_this_time, dim_head}});
    k_trans.Resize(
        {{bsz, num_head, max_enc_len_this_time + pre_cache_length, dim_head}});
    v_trans.Resize(
        {{bsz, num_head, max_enc_len_this_time + pre_cache_length, dim_head}});
    qktv_out.Resize({{bsz, num_head, max_enc_len_this_time, dim_head}});

    dev_ctx.template Alloc<T>(&q_trans, q_trans.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&k_trans, k_trans.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&v_trans, v_trans.numel() * sizeof(T));
    dev_ctx.template Alloc<T>(&qktv_out, qktv_out.numel() * sizeof(T));
  }

  // qkv: [self.batch_size * self.seq_len, nd*hd]
  qkv_buf = qkv;
  *key_cache_out = key_cache;
  *value_cache_out = value_cache;

  if (max_enc_len_this_time > 0) {
    const int* sequence_lengths_data = seq_lens_encoder.data<int>();
    if (!pre_key_cache) {
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
                                      &fmha_buf,
                                      &softmax_out,
                                      &softmax_lse,
                                      &seed_offset);
      VLOG(3) << "-------fmha_buf dims: " << fmha_buf.dims();
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

      phi::FlashAttnUnpaddedKernel<T>(dev_ctx,
                                q_trans,
                                k_trans,
                                v_trans,
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
                                &fmha_buf,
                                &softmax_out,
                                &softmax_lse,
                                &seed_offset);
    }
  }

  if (max_dec_len_this_time > 0) {
    const int* sequence_lengths_data = seq_lens_decoder.data<int>();

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

    // 写一个 kernel，更新 cache_k, cache_v
    // 更新 k，将 unpadding_k 拼接到 key_cache 后面
    phi::DenseTensor key_cache_transed =
        phi::Transpose<T, phi::GPUContext>(dev_ctx, key_cache, {0, 2, 1, 3});
    phi::DenseTensor value_cache_transed =
        phi::Transpose<T, phi::GPUContext>(dev_ctx, value_cache, {0, 2, 1, 3});
    CacheKernel<T>(dev_ctx,
      unpadding_k,
      unpadding_v,
      key_cache_transed,
      value_cache_transed,
      unpadding_k_after_cache,
      unpadding_v_after_cache,
      token_num_in_cache,
      bsz,
      token_num,
      num_head,
      dim_head);
    VLOG(3) << "-------unpadding_q dims: " << unpadding_q.dims();
    VLOG(3) << "-------unpadding_k dims: " << unpadding_k.dims();
    VLOG(3) << "-------unpadding_v dims: " << unpadding_v.dims();
    VLOG(3) << "-------unpadding_k_after_cache dims: " << unpadding_k_after_cache.dims();
    VLOG(3) << "-------unpadding_v_after_cache dims: " << unpadding_v_after_cache.dims();

    VLOG(3) << "-------cu_seqlens_q dims: " << cu_seqlens_q.dims();
    VLOG(3) << "-------cu_seqlens_k dims: " << cu_seqlens_k.dims();
    VLOG(3) << "-------max_dec_len_this_time: " << max_dec_len_this_time;
    // paddle::funcs::TensorFormatter formatter;
    // formatter.Print(unpadding_q, "unpadding_q");
    // formatter.Print(unpadding_k_after_cache, "unpadding_k_after_cache");
    // formatter.Print(unpadding_v_after_cache, "unpadding_v_after_cache");
    // formatter.Print(key_cache, "key_cache");
    // formatter.Print(value_cache, "value_cache");

    phi::FlashAttnUnpaddedKernel<T>(dev_ctx,
                                    unpadding_q,
                                    unpadding_k_after_cache,
                                    unpadding_v_after_cache,
                                    cu_seqlens_q,
                                    cu_seqlens_k,
                                    paddle::none /*fixed_seed_offset*/,
                                    causual ? paddle::none : mask,
                                    max_dec_len_this_time,
                                    max_dec_len_this_time,
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
    // formatter.Print(fmha_buf, "fmha_buf");
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
