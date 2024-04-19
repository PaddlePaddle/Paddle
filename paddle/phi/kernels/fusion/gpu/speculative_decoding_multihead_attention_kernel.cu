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

// Put the first cur_token_num tokens from kv_cache and unpadding_kv to
// unpadding_kv_after_cache.
template <typename T, int VecSize = 1>
__global__ void write_cache_to_unpadding_kv_kernel(
    const T *__restrict__ unpadding_k,  // [cur_num_tokens, num_heads, dim_head]
    const T *__restrict__ unpadding_v,  // [cur_num_tokens, num_heads, dim_head]
    const T *__restrict__ key_cache,    // [1, max_seq_len, num_head, dim_head]
    const T *__restrict__ value_cache,  // [1, max_seq_len, num_head, dim_head]
    T *__restrict__ unpadding_k_after_cache,  // [cur_token_num +
                                              // token_num_in_cache, num_head,
                                              // dim_head]
    T *__restrict__ unpadding_v_after_cache,  // [cur_token_num +
                                              // token_num_in_cache, num_head,
                                              // dim_head]
    const int token_num_in_cache,  // [bsz]，valid token_num in cache
    const int
        cur_num_tokens_k,  // Token_num of current key. In speculative decoding,
                           // token_num_k > token_num_q in decoder phase.
    const int num_heads,
    const int dim_head) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT unpadding_k_after_cache_vec;
  LoadT unpadding_v_after_cache_vec;

  int64_t idx = (blockDim.x * blockIdx.x + threadIdx.x) * VecSize;
  int stride = blockDim.x * gridDim.x * VecSize;

  const int64_t hidden_size = num_heads * dim_head;
  const int64_t offset = cur_num_tokens_k * hidden_size;
  const int64_t cur_hidden_size = token_num_in_cache * hidden_size;

  for (; idx < cur_hidden_size; idx += stride) {
    phi::Load<T, VecSize>(&key_cache[idx], &unpadding_k_after_cache_vec);
    phi::Load<T, VecSize>(&value_cache[idx], &unpadding_v_after_cache_vec);

    phi::Store<T, VecSize>(unpadding_k_after_cache_vec,
                           &unpadding_k_after_cache[idx]);
    phi::Store<T, VecSize>(unpadding_v_after_cache_vec,
                           &unpadding_v_after_cache[idx]);
  }

  for (; cur_hidden_size <= idx && idx < cur_hidden_size + offset;
       idx += stride) {
    phi::Load<T, VecSize>(&unpadding_k[idx - cur_hidden_size],
                          &unpadding_k_after_cache_vec);
    phi::Load<T, VecSize>(&unpadding_v[idx - cur_hidden_size],
                          &unpadding_v_after_cache_vec);

    phi::Store<T, VecSize>(unpadding_k_after_cache_vec,
                           &unpadding_k_after_cache[idx]);
    phi::Store<T, VecSize>(unpadding_v_after_cache_vec,
                           &unpadding_v_after_cache[idx]);
  }
}

// Put the first cur_token_num tokens from kv_cache and unpadding_kv to
// unpadding_kv_after_cache.
template <typename T>
void WriteCacheToKVKernel(
    const phi::GPUContext &dev_ctx,
    const phi::DenseTensor &unpadding_k,  // [cur_token_num, num_head, dim_head]
    const phi::DenseTensor &unpadding_v,  // [cur_token_num, num_head, head_dim]
    const phi::DenseTensor &key_cache,    // [bsz(1), max_seq_len, num_head,
                                          // dim_head](has been transposed.)
    const phi::DenseTensor &value_cache,
    phi::DenseTensor *unpadding_k_after_cache,
    phi::DenseTensor *unpadding_v_after_cache,
    const int token_num_in_cache,  // the token_num in cache that are valid.
    const int batch_size,
    const int num_tokens,
    const int num_heads,
    const int head_size) {
  typedef PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  // stage 1: write qkv to cache [pre_cache_length:]
  int elem_nums = unpadding_k_after_cache->numel();  // just k and v
  constexpr int PackSize = 16 / sizeof(T);
  int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);

  write_cache_to_unpadding_kv_kernel<DataType_, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          reinterpret_cast<DataType_ *>(const_cast<T *>(unpadding_k.data<T>())),
          reinterpret_cast<DataType_ *>(const_cast<T *>(unpadding_v.data<T>())),
          reinterpret_cast<DataType_ *>(const_cast<T *>(key_cache.data<T>())),
          reinterpret_cast<DataType_ *>(const_cast<T *>(value_cache.data<T>())),
          reinterpret_cast<DataType_ *>(unpadding_k_after_cache->data<T>()),
          reinterpret_cast<DataType_ *>(unpadding_v_after_cache->data<T>()),
          token_num_in_cache,
          num_tokens,
          num_heads,
          head_size);
}

// Used in speculative decoding
template <typename T, int VecSize = 1>
__global__ void NeoxVariableLengthRotarySpecuKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int token_num_in_cahce,
    const int last_dim) {
  // [token_num, 2, num_head, dim_head / 2]
  using LoadT = phi::AlignedVector<T, VecSize>;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT left_vec;
  LoadT right_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * half_lastdim;
  const int full_hidden_size = num_head * last_dim;
  const int offset = 2 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx;
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / half_lastdim;
    const int h_bias = qkv_bias % half_lastdim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx = (ori_seq_id + token_num_in_cahce) * last_dim + h_bias;
    const int base_idx_left = token_idx * 3 * full_hidden_size +
                              qkv_id * full_hidden_size + hi * last_dim +
                              h_bias;
    const int base_idx_right = base_idx_left + half_lastdim;

    phi::Load<T, VecSize>(&qkv[base_idx_left], &left_vec);
    phi::Load<T, VecSize>(&qkv[base_idx_right], &right_vec);
    phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      const float input_left = static_cast<float>(left_vec[i]);
      const float input_right = static_cast<float>(right_vec[i]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      left_vec[i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      right_vec[i] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    phi::Store<T, VecSize>(left_vec, &qkv_out[base_idx_left]);
    phi::Store<T, VecSize>(right_vec, &qkv_out[base_idx_right]);
  }
}

// used in the decoder phase of speculative decoding
template <typename T, int VecSize = 1>
__global__ void VariableLengthRotarySpecuKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int token_num_in_cahce,
    const int last_dim) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 2 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens && seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int emb_idx =
        (ori_seq_id + token_num_in_cahce) * half_lastdim + h_bias / 2;
    const int64_t base_idx = token_idx * 3 * hidden_size +
                             qkv_id * hidden_size + hi * last_dim + h_bias;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = static_cast<float>(src_vec[2 * i]);
      const float input_right = static_cast<float>(src_vec[2 * i + 1]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      src_vec[2 * i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      src_vec[2 * i + 1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

// used in the decoder phase of speculative decoding
// NOTE: when use this kernel, please set use_neox_style = true.
template <typename T>
void rotary_qk_variable_specu(
    const phi::GPUContext &dev_ctx,
    T *qkv,                   // [token_num, 3, num_head, dim_head]
    const T *qkv_input,       // qkv
    const float *rotary_emb,  // [2, 1, seq_len, 1, dim_head]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int token_num_in_cache,
    const int input_output_len,
    const int dim_head,
    bool use_neox_style = false) {
  int elem_nums = token_num * 2 * head_num * dim_head;  // just q and k
  if (use_neox_style) {
    elem_nums = token_num * head_num * dim_head;
  }
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    VariableLengthRotarySpecuKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv,
                                                        elem_nums,
                                                        head_num,
                                                        seq_len,
                                                        token_num_in_cache,
                                                        dim_head);
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    NeoxVariableLengthRotarySpecuKernel<T, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                        cos_emb,
                                                        sin_emb,
                                                        padding_offsets,
                                                        seq_lens,
                                                        qkv,
                                                        elem_nums,
                                                        head_num,
                                                        seq_len,
                                                        token_num_in_cache,
                                                        dim_head);
  }
}

template <typename T, typename Context>
void SpeculativeDecodingMultiheadAttentionKernel(
    const Context &dev_ctx,
    const DenseTensor &qkv,        // [token_num, 3*hidden_dim]
    const DenseTensor &key_cache,  // [bsz, num_head, max_seq_len, dim_head]
    const DenseTensor &value_cache,
    const DenseTensor &seq_lens_encoder,
    const DenseTensor &seq_lens_decoder,
    const DenseTensor &seq_lens_this_time,
    const DenseTensor &padding_offsets,
    const DenseTensor &cum_offsets,
    const DenseTensor &cu_seqlens_q,
    const DenseTensor &cu_seqlens_k,
    const paddle::optional<DenseTensor> &rope_emb,
    const paddle::optional<DenseTensor> &mask,
    const paddle::optional<DenseTensor> &qkv_bias,
    const int max_enc_len_this_time,
    const int max_dec_len_this_time,
    const int token_num_in_cache,
    int max_seq_len,
    bool use_neox_style,
    const std::string &compute_dtype,
    DenseTensor *fmha_out,
    DenseTensor *qkv_out,
    DenseTensor *key_cache_out,
    DenseTensor *value_cache_out) {
  phi::DenseTensor qkv_buf;
  phi::DenseTensor fmha_buf;

  VLOG(1) << "fmha_out " << fmha_out->dims();
  dev_ctx.template Alloc<T>(fmha_out);
  fmha_buf = *fmha_out;

  InitValue(dev_ctx, fmha_buf.data<T>(), fmha_buf.numel(), static_cast<T>(0.));
  const auto &input_dims = qkv.dims();
  const auto &key_cache_dims = key_cache.dims();
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
  auto *max_dec_len_data = dev_ctx.template Alloc<int>(
      &max_dec_len_tensor, max_dec_len_tensor.numel() * sizeof(int));

  phi::DenseTensor max_enc_len_tensor;
  max_enc_len_tensor.Resize({{1}});
  auto *max_enc_len_data = dev_ctx.template Alloc<int>(
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
    const int *sequence_lengths_data = seq_lens_encoder.data<int>();
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
    const int *sequence_lengths_data = seq_lens_decoder.data<int>();
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
                            &unpadding_k_after_cache,
                            &unpadding_v_after_cache,
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
