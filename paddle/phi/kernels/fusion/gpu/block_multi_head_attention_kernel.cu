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
#include "paddle/phi/kernels/fusion/cutlass/variable_length_memory_efficient_attention.h"
#include "paddle/phi/kernels/fusion/gpu/block_attn.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/utils/none.h"
#include "paddle/phi/core/flags.h"
#include <fstream>


PHI_DECLARE_int32(max_enc_len_this_time_data);
PHI_DECLARE_int32(max_dec_len_this_time_data);

namespace phi {

template<typename T>
void PrintTensor(const DenseTensor& t, int num) {
  // VLOG(2) << "  - place: " << t.place() << "\n";
  // VLOG(2) << "  - shape: [" << t.dims() << "]\n";

  // std::vector<T> data(num);

  // cudaMemcpy(data.data(), t.data<T>(), num * sizeof(T), cudaMemcpyDeviceToHost);

  // std::string res;
  // for (int i = 0; i < num; ++i) {
  //   res += (std::to_string(static_cast<float>(data[i])) + " ");
  // }
  // VLOG(2) << res;
  
}
namespace fusion {

template <typename data_t>
inline HOSTDEVICE data_t RoundWithTiesToEven(data_t x) {
  data_t xLower = floor(x);
  data_t xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  data_t dLower = x - xLower;
  data_t dUpper = xUpper - x;
  return static_cast<data_t>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename T>
__forceinline__ __device__ T add_mul(T a, T b, T c) {
    return (a + b) * c;
}

template<>
__forceinline__ __device__ half add_mul<half>(half a, half b, half c) {
    return __hmul(__hadd(a, b), c);
}

template<>
__forceinline__ __device__ __nv_bfloat16 add_mul<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    return __hmul(__hadd(a, b), c);
}



template <typename data_t>
__forceinline__ __device__ int8_t quant_helper(const data_t input,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);

  if (round_type == 0) {
    quant_value = static_cast<float>(RoundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename data_t>
__forceinline__ __device__ int8_t quant_helper(const data_t input,
                                               const data_t shift,
                                               const data_t smooth,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  auto smooth_out = add_mul(input, shift, smooth);
  float quant_value = max_bound * scale * static_cast<float>(smooth_out);

  if (round_type == 0) {
    quant_value = static_cast<float>(RoundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename data_t>
__global__ void QuantKernel(const data_t* input,
                            char4* output,
                            const float scale,
                            const int m,
                            const int n,
                            const int round_type,
                            const float max_bound,
                            const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;
  bool check = ((m_id < m) && (n_id < n));

  if (check) {
    char4 tmp;
    tmp.x = quant_helper(
        input[m_id * n + n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper(
        input[m_id * n + n_id + 1], scale, round_type, max_bound, min_bound);
    tmp.z = quant_helper(
        input[m_id * n + n_id + 2], scale, round_type, max_bound, min_bound);
    tmp.w = quant_helper(
        input[m_id * n + n_id + 3], scale, round_type, max_bound, min_bound);

    output[(m_id * n + n_id) >> 2] = tmp;
  }
}

template <typename data_t>
__global__ void QuantKernel(const data_t* input,
                            const data_t* shift,
                            const data_t* smooth,
                            char4* output,
                            const float scale,
                            const int m,
                            const int n,
                            const int round_type,
                            const float max_bound,
                            const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;
  bool check = ((m_id < m) && (n_id < n));

  if (check) {
    char4 tmp;
    tmp.x = quant_helper(
        input[m_id * n + n_id], shift[n_id], smooth[n_id], scale, round_type, max_bound, min_bound);
    tmp.y = quant_helper(
        input[m_id * n + n_id + 1], shift[n_id + 1], smooth[n_id + 1], scale, round_type, max_bound, min_bound);
    tmp.z = quant_helper(
        input[m_id * n + n_id + 2], shift[n_id + 2], smooth[n_id + 2], scale, round_type, max_bound, min_bound);
    tmp.w = quant_helper(
        input[m_id * n + n_id + 3], shift[n_id + 3], smooth[n_id + 3], scale, round_type, max_bound, min_bound);

    output[(m_id * n + n_id) >> 2] = tmp;
  }
}


template <typename T, int VecSize>
__global__ void DequantKernel(T* output,
                              const int32_t* input,
                              const int64_t m,  // batch size
                              const int64_t n,  // hidden
                              const float* dequant_out_scale_data) {
  int64_t numel = m * n;
  int64_t stride = blockDim.x * gridDim.x * VecSize;
  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int64_t col_id = idx % n;

  phi::AlignedVector<int32_t, VecSize> in_vec;
  phi::AlignedVector<float, VecSize> out_scale_vec;
  phi::AlignedVector<T, VecSize> out_vec;

  for (; idx < numel; idx += stride) {
    phi::Load<int32_t, VecSize>(input + idx, &in_vec);
    phi::Load<float, VecSize>(dequant_out_scale_data + col_id, &out_scale_vec);

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      out_vec[i] =
          static_cast<T>(static_cast<float>(in_vec[i]) * out_scale_vec[i]);
    }

    phi::Store<T, VecSize>(out_vec, output + idx);
  }
}

template <typename T, typename Context>
void DispatchWithDtype(
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
    const paddle::optional<DenseTensor>& cache_k_quant_scales,
    const paddle::optional<DenseTensor>& cache_v_quant_scales,
    const paddle::optional<DenseTensor>& cache_k_dequant_scales,
    const paddle::optional<DenseTensor>& cache_v_dequant_scales,
    const paddle::optional<DenseTensor> &qkv_out_scale,
    const paddle::optional<DenseTensor> &qkv_bias,
    const paddle::optional<DenseTensor> &out_shift,
    const paddle::optional<DenseTensor> &out_smooth,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    const bool dynamic_cachekv_quant,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_scale,
    const std::string& compute_dtype,
    DenseTensor* fmha_out,
    DenseTensor* qkv_out,
    DenseTensor* key_cache_out,
    DenseTensor* value_cache_out) {

  phi::DenseTensor qkv_buf;
  phi::DenseTensor fmha_buf;

  VLOG(1) << "fmha_out " << fmha_out->dims();
  if (out_scale <= 0) {
    dev_ctx.template Alloc<T>(fmha_out);
    fmha_buf = *fmha_out; 
  } else {
    fmha_buf.Resize(fmha_out->dims());
    dev_ctx.template Alloc<T>(&fmha_buf);
    dev_ctx.template Alloc<int8_t>(fmha_out);
  }

  InitValue(
      dev_ctx, fmha_buf.data<T>(), fmha_buf.numel(), static_cast<T>(0.));
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

  bool use_pre_cache = false;
  int pre_cache_length = 0;
  if (pre_key_cache) {
    use_pre_cache = true;
    pre_cache_length = pre_key_cache.get().dims()[2];
  }
  VLOG(1) << "token_num: " << token_num
          << " pre_cache_length: " << pre_cache_length;

  // phi::DenseTensor max_dec_len_tensor;
  // max_dec_len_tensor.Resize({{1}});
  // auto* max_dec_len_data = dev_ctx.template Alloc<int>(
  //     &max_dec_len_tensor, max_dec_len_tensor.numel() * sizeof(int));
  // int max_dec_len_this_time =
  //     GetMaxLen(dev_ctx, seq_lens_decoder, &max_dec_len_tensor, bsz);

  // phi::DenseTensor max_enc_len_tensor;
  // max_enc_len_tensor.Resize({{1}});
  // auto* max_enc_len_data = dev_ctx.template Alloc<int>(
  //     &max_enc_len_tensor, max_enc_len_tensor.numel() * sizeof(int));
  // int max_enc_len_this_time =
  //     GetMaxLen(dev_ctx, seq_lens_encoder, &max_enc_len_tensor, bsz);

 

  int max_enc_len_this_time = 0;
  int max_dec_len_this_time = 0;

  std::ifstream infile("max_len.txt", std::ios::in);
  infile >> max_enc_len_this_time >> max_dec_len_this_time;
  infile.close();


  phi::DenseTensor qkv_out_decoder;
  if (max_dec_len_this_time > 0) {
    qkv_out_decoder.Resize({{bsz, 3, num_head, dim_head}});
    auto* qkv_out_decoder_data = dev_ctx.template Alloc<T>(
        &qkv_out_decoder, qkv_out_decoder.numel() * sizeof(T));
  }
  VLOG(1) << "max_len end";
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
  VLOG(1) << "encoder";
  VLOG(1) << "max_enc_len_this_time: " << max_enc_len_this_time;


  // Begin to compute 
  
  if (qkv_out_scale) {
    VLOG(1) << "qkv_out_scale: " << qkv_out_scale.get_ptr()->dims();
    qkv_buf.Resize(qkv.dims());
    dev_ctx.template Alloc<T>(&qkv_buf, qkv_buf.numel() * sizeof(T));

    int64_t numel = qkv.numel();
    constexpr int64_t thread_per_block = 512;
    constexpr int DequantKernelVecSize = 4;
    int64_t block_per_grid = (numel / DequantKernelVecSize + thread_per_block - 1) / thread_per_block;
    PrintTensor<int32_t>(qkv, 10);
    DequantKernel<T, DequantKernelVecSize><<<block_per_grid, thread_per_block, 0, dev_ctx.stream()>>>(qkv_buf.data<T>(), qkv.data<int32_t>(), input_dims[0], input_dims[1], qkv_out_scale.get_ptr()->data<float>());
    PrintTensor<T>(qkv_buf, 10);
  } else {
    VLOG(1) << "qkv_out_scale is none";
    qkv_buf = qkv;
  }

  if (qkv_bias) {
    VLOG(1) << "has bias";
    PrintTensor<T>(qkv_bias.get(), 10);
    std::vector<const phi::DenseTensor*> ins = {&qkv_buf, qkv_bias.get_ptr()};
    std::vector<phi::DenseTensor*> outs = {&qkv_buf};
    phi::funcs::BroadcastKernel<T>(
        dev_ctx, ins, &outs, phi::funcs::AddFunctor<T>());
  }

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
      
      VLOG(1) << "rope end";
      PrintTensor<T>(qkv_buf, 10);                   
    }
    
    VLOG(1) << "causual: " << causual;
    if (!use_pre_cache) {
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
      VLOG(1) << "qkv split end";
      PrintTensor<T>(unpadding_q, 10); 
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
    } else {
      qkv_transpose_split<T>(
          dev_ctx,
          q_trans.data<T>(),
          k_trans.data<T>(),
          v_trans.data<T>(),
          qkv_buf.data<T>(),
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
      InvokeTransposeRemovePadding<T>(dev_ctx,
                                      qktv_out.data<T>(),
                                      sequence_lengths_data,
                                      fmha_buf.data<T>(),
                                      bsz,
                                      num_head,
                                      max_enc_len_this_time,
                                      max_seq_len,
                                      dim_head,
                                      token_num,
                                      padding_offsets.data<int>());
    }
    VLOG(1) << "flash end";
    if (cache_k_quant_scales && dynamic_cachekv_quant) {
      DynamicQuantCacheKernel<T>(dev_ctx,
                                 qkv_buf,
                                 block_tables,
                                 padding_offsets,
                                 seq_lens_encoder,
                                 *(cache_k_quant_scales.get_ptr()),
                                 *(cache_v_quant_scales.get_ptr()),
                                 *(cache_k_dequant_scales.get_ptr()),
                                 *(cache_v_dequant_scales.get_ptr()),
                                 pre_key_cache,
                                 pre_value_cache,
                                 bsz,
                                 num_head,
                                 dim_head,
                                 max_seq_len,
                                 pre_cache_length,
                                 key_cache_out,
                                 value_cache_out);
    } else {
      CacheKernel<T>(dev_ctx,
                     qkv_buf,
                     block_tables,
                     padding_offsets,
                     seq_lens_encoder,
                     pre_key_cache,
                     pre_value_cache,
                     cache_k_quant_scales,
                     cache_v_quant_scales,
                     bsz,
                     token_num,
                     num_head,
                     dim_head,
                     max_seq_len,
                     pre_cache_length,
                     key_cache_out,
                     value_cache_out);
    }
    VLOG(1) << "cache end";
  }
  VLOG(1) << "encoder done";
  VLOG(1) << "max_dec_len_this_time: " << max_dec_len_this_time;
  if (max_dec_len_this_time > 0) {
    GetDecoderTensor<T>(dev_ctx,
                        qkv_buf,
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
    blha<T>(
        dev_ctx,
        qkv_out_decoder,
        nullptr,  // qkv_bias
        &block_tables,
        tgt_mask ? &tgt_mask.get() : nullptr,
        &cum_offsets,
        &seq_lens_decoder,
        rope_emb ? &rope_emb.get() : nullptr,  // rope_emb
        key_cache_out,
        value_cache_out,
        &fmha_buf,
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
        use_neox_style,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
        cache_k_quant_scales ? cache_k_quant_scales.get_ptr() : nullptr,
        cache_v_quant_scales ? cache_v_quant_scales.get_ptr() : nullptr,
        cache_k_dequant_scales ? cache_k_dequant_scales.get_ptr() : nullptr,
        cache_v_dequant_scales ? cache_v_dequant_scales.get_ptr() : nullptr);
    VLOG(1) << "blha end";
  }
  
  if (out_scale > 0) {
    int m = fmha_out->dims()[0];
    int n = fmha_out->dims()[1];
    dim3 grid((n >> 2 + 31) / 32, (m + 31) / 32);
    dim3 block(32, 32);
    PrintTensor<T>(fmha_buf, 10);
    if (out_shift && out_smooth) {
      QuantKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(fmha_buf.data<T>(), 
                                                           out_shift.get_ptr()->data<T>(), 
                                                           out_smooth.get_ptr()->data<T>(), 
                                                           reinterpret_cast<char4*>(fmha_out->data<int8_t>()), 
                                                           out_scale,
                                                           m,
                                                           n,
                                                           quant_round_type,
                                                           quant_max_bound,
                                                           quant_min_bound
                                                           );
    } else {
      QuantKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(fmha_buf.data<T>(), 
                                                           reinterpret_cast<char4*>(fmha_out->data<int8_t>()), 
                                                           out_scale,
                                                           m,
                                                           n,
                                                           quant_round_type,
                                                           quant_max_bound,
                                                           quant_min_bound
                                                           );
    }
    PrintTensor<int8_t>(*fmha_out, 10);
  }
  VLOG(1) << "decoder done";
}


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
    const paddle::optional<DenseTensor>& cache_k_quant_scales,
    const paddle::optional<DenseTensor>& cache_v_quant_scales,
    const paddle::optional<DenseTensor>& cache_k_dequant_scales,
    const paddle::optional<DenseTensor>& cache_v_dequant_scales,
    const paddle::optional<DenseTensor> &qkv_out_scale,
    const paddle::optional<DenseTensor> &qkv_bias,
    const paddle::optional<DenseTensor> &out_shift,
    const paddle::optional<DenseTensor> &out_smooth,
    int max_seq_len,
    int block_size,
    bool use_neox_style,
    const bool dynamic_cachekv_quant,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_scale,
    const std::string& compute_dtype,
    DenseTensor* fmha_out,
    DenseTensor* qkv_out,
    DenseTensor* key_cache_out,
    DenseTensor* value_cache_out) {
  VLOG(1) << "compute_dtype " << compute_dtype;
  VLOG(1) << "qkv.dtype() " << qkv.dtype();
  if (qkv.dtype() == phi::DataType::INT32) {
    VLOG(1) << "qkv.dtype() int32";
    if (compute_dtype == "fp16") {
        VLOG(1) << "compute_dtype fp16";
        DispatchWithDtype<phi::dtype::float16, Context>(dev_ctx,
    qkv,
    key_cache,
    value_cache,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    padding_offsets,
    cum_offsets,
    cu_seqlens_q,
    cu_seqlens_k,
    block_tables,
    pre_key_cache,
    pre_value_cache,
    rope_emb,
    mask,
    tgt_mask,
    cache_k_quant_scales,
    cache_v_quant_scales,
    cache_k_dequant_scales,
    cache_v_dequant_scales,
    qkv_out_scale,
    qkv_bias,
    out_shift,
    out_smooth,
    max_seq_len,
    block_size,
    use_neox_style,
    dynamic_cachekv_quant,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
    out_scale,
    compute_dtype,
    fmha_out,
    qkv_out,
    key_cache_out,
    value_cache_out);
    } else if (compute_dtype == "bf16") {
        DispatchWithDtype<phi::dtype::bfloat16, Context>(dev_ctx,
    qkv,
    key_cache,
    value_cache,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    padding_offsets,
    cum_offsets,
    cu_seqlens_q,
    cu_seqlens_k,
    block_tables,
    pre_key_cache,
    pre_value_cache,
    rope_emb,
    mask,
    tgt_mask,
    cache_k_quant_scales,
    cache_v_quant_scales,
    cache_k_dequant_scales,
    cache_v_dequant_scales,
    qkv_out_scale,
    qkv_bias,
    out_shift,
    out_smooth,
    max_seq_len,
    block_size,
    use_neox_style,
    dynamic_cachekv_quant,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
    out_scale,
    compute_dtype,
    fmha_out,
    qkv_out,
    key_cache_out,
    value_cache_out);
    }
  } else {
    VLOG(1) << "qkv.dtype() NOT int32";
    if (std::is_same<T, phi::dtype::float16>::value) {
        DispatchWithDtype<phi::dtype::float16, Context>(dev_ctx,
    qkv,
    key_cache,
    value_cache,
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    padding_offsets,
    cum_offsets,
    cu_seqlens_q,
    cu_seqlens_k,
    block_tables,
    pre_key_cache,
    pre_value_cache,
    rope_emb,
    mask,
    tgt_mask,
    cache_k_quant_scales,
    cache_v_quant_scales,
    cache_k_dequant_scales,
    cache_v_dequant_scales,
    qkv_out_scale,
    qkv_bias,
    out_shift,
    out_smooth,
    max_seq_len,
    block_size,
    use_neox_style,
    dynamic_cachekv_quant,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
    out_scale,
    compute_dtype,
    fmha_out,
    qkv_out,
    key_cache_out,
    value_cache_out);
    } else if (std::is_same<T, phi::dtype::float16>::value) {
        DispatchWithDtype<phi::dtype::bfloat16, Context>(dev_ctx,
                  qkv,
                  key_cache,
                  value_cache,
                  seq_lens_encoder,
                  seq_lens_decoder,
                  seq_lens_this_time,
                  padding_offsets,
                  cum_offsets,
                  cu_seqlens_q,
                  cu_seqlens_k,
                  block_tables,
                  pre_key_cache,
                  pre_value_cache,
                  rope_emb,
                  mask,
                  tgt_mask,
                  cache_k_quant_scales,
                  cache_v_quant_scales,
                  cache_k_dequant_scales,
                  cache_v_dequant_scales,
                  qkv_out_scale,
                  qkv_bias,
                  out_shift,
                  out_smooth,
                  max_seq_len,
                  block_size,
                  use_neox_style,
                  dynamic_cachekv_quant,
                  quant_round_type,
                  quant_max_bound,
                  quant_min_bound,
                  out_scale,
                  compute_dtype,
                  fmha_out,
                  qkv_out,
                  key_cache_out,
                  value_cache_out);
    }
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(block_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::BlockMultiheadAttentionKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int32_t) {}
