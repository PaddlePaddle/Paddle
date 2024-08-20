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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/fusion/gpu/block_attn_tc_utils.h"

namespace phi {
namespace fusion {

template <typename CacheKV_traits, typename T, int kNThreads>
__global__ __launch_bounds__(kNThreads) void write_encoder_cache_kv_kernel(
    const T* qkv_out,
    const int* cu_seqlens_q,
    const int* block_tables,
    const int* seq_lens,
    uint8_t* cache_k,
    uint8_t* cache_v,
    const T* cache_k_quant_scale,
    const T* cache_v_quant_scale,
    const int num_head,
    const int num_kv_head,
    const int head_dim,
    const int max_blocks_per_seq) {
  using index_t = typename CacheKV_traits::cache_index;
  using cuteType = typename CacheKV_traits::cuteType;
  constexpr int kHeadDim = CacheKV_traits::kHeadDim;
  constexpr int kBlockSize = CacheKV_traits::kBlockSize;
  constexpr int kDataBits = CacheKV_traits::kDataBits;
  const int tidx = threadIdx.x;
  const int block_idx = blockIdx.x;
  const int bi = blockIdx.y;
  const int head_idx = blockIdx.z;
  const int seq_len = seq_lens[bi];
  if (seq_len == 0) return;
  const int blocks = (seq_len + kBlockSize - 1) / kBlockSize;
  if (block_idx >= blocks) return;

  constexpr int32_t cache_headdim = kDataBits == 4 ? kHeadDim / 2 : kHeadDim;

  const int qkv_size_per_token = (num_head + 2 * num_kv_head) * kHeadDim;
  const int q_seq_idx = cu_seqlens_q[bi] + block_idx * kBlockSize;
  const int k_offset =
      q_seq_idx * qkv_size_per_token + (head_idx + num_head) * kHeadDim;
  const int v_offset = k_offset + num_kv_head * kHeadDim;
  const int remain_seq_len = seq_len - block_idx * kBlockSize;

  Tensor gK = make_tensor(
      make_gmem_ptr(reinterpret_cast<const cuteType*>(qkv_out) + k_offset),
      Shape<Int<kBlockSize>, Int<kHeadDim>>{},
      make_stride(qkv_size_per_token, _1{}));

  Tensor gV = make_tensor(
      make_gmem_ptr(reinterpret_cast<const cuteType*>(qkv_out) + v_offset),
      Shape<Int<kBlockSize>, Int<kHeadDim>>{},
      make_stride(qkv_size_per_token, _1{}));

  // Shared memory.
  extern __shared__ char smem_[];
  Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<cuteType*>(smem_)),
                          typename CacheKV_traits::SmemLayoutKV{});

  Tensor sV = make_tensor(sK.data() + size(sK),
                          typename CacheKV_traits::SmemLayoutKV{});

  Tensor sVt =
      make_tensor(sV.data(), typename CacheKV_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data(), typename CacheKV_traits::SmemLayoutVtransposedNoSwizzle{});

  auto gmem_thr_copy_KV =
      typename CacheKV_traits::GmemTiledCopyQ{}.get_thread_slice(tidx);

  Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);
  Tensor cKV = make_identity_tensor(make_shape(kBlockSize, kHeadDim));
  Tensor tKVcKV = gmem_thr_copy_KV.partition_S(cKV);

  if (remain_seq_len < kBlockSize) {
    block_attn::copy<false>(
        gmem_thr_copy_KV, tKgK, tKsK, tKVcKV, remain_seq_len);
  } else {
    block_attn::copy(gmem_thr_copy_KV, tKgK, tKsK, tKVcKV);
  }

  const int col = tidx % 4;
  const int row = tidx / 4;

  using T_vec_4 = typename block_attn::Vec<cuteType, 4>;
  using T_vec_32 = typename block_attn::Vec<cuteType, 32>;

  const T_vec_32 scale_k = *reinterpret_cast<const T_vec_32*>(
      cache_k_quant_scale + head_idx * kHeadDim + col * 32);
  const T_vec_4 scale_v = *reinterpret_cast<const T_vec_4*>(
      cache_v_quant_scale + head_idx * kHeadDim + row * 4);

  cute::cp_async_fence();
  typename CacheKV_traits::TiledMma tiled_mma;

  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  auto smem_thr_copy_K =
      make_tiled_copy_B(typename CacheKV_traits::SmemCopyAtom{}, tiled_mma)
          .get_thread_slice(tidx);
  auto smem_thr_copy_V =
      make_tiled_copy_B(typename CacheKV_traits::SmemCopyAtomTransposed{},
                        tiled_mma)
          .get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

  Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
  Tensor tSrV_copy_view = smem_thr_copy_V.retile_D(tOrVt);
  block_attn::cp_async_wait<0>();

  __syncthreads();
  if (remain_seq_len < kBlockSize) {
    block_attn::copy<false>(
        gmem_thr_copy_KV, tVgV, tVsV, tKVcKV, remain_seq_len);
  } else {
    block_attn::copy(gmem_thr_copy_KV, tVgV, tVsV, tKVcKV);
  }
  cute::cp_async_fence();

  constexpr int cache_data_num = size<2>(tSsK) * kNThreads * (kDataBits / 4);
  __shared__ uint32_t kv[cache_data_num];

  using uint_vec_4 = typename block_attn::Vec<uint32_t, 4>;
  using uint_vec_2 = typename block_attn::Vec<uint32_t, 2>;

#pragma unroll
  for (int i = 0; i < size<2>(tSsK); i++) {
    copy(smem_thr_copy_K, tSsK(_, _, i), tSrK_copy_view(_, _, i));
    block_attn::write_cache_k<cuteType, kDataBits, kNThreads>(
        tSrK(_, 0, i).data(),
        tSrK(_, 1, i).data(),
        kv,
        scale_k.data.elt + i * 4,
        tidx,
        i);
  }

  const int32_t* block_table_now = block_tables + bi * max_blocks_per_seq;
  const uint32_t physical_block_number = block_table_now[block_idx];

  const index_t cache_offset =
      physical_block_number * num_kv_head * kBlockSize * cache_headdim +
      head_idx * kBlockSize * cache_headdim;

  constexpr int pack_size = 16 / sizeof(uint32_t);
  using pack_4_uint_vec = block_attn::Vec<uint32_t, pack_size>;
  pack_4_uint_vec* cache_k_cur =
      reinterpret_cast<pack_4_uint_vec*>(cache_k + cache_offset);

  block_attn::cp_async_wait<0>();
  __syncthreads();

  pack_4_uint_vec* smem_pack_4 = reinterpret_cast<pack_4_uint_vec*>(kv);

#pragma unroll
  for (int i = tidx; i < cache_data_num / pack_size; i += kNThreads) {
    cache_k_cur[i] = smem_pack_4[i];
  }
  __syncthreads();

  uint_vec_4 value_c8;
  uint_vec_2 value_c4;

#pragma unroll
  for (int i = 0; i < size<2>(tOrVt); i++) {
    copy(smem_thr_copy_V, tOsVt(_, _, i), tSrV_copy_view(_, _, i));
#pragma unroll
    for (int j = 0; j < size<1>(tOrVt); j += 2) {
      if constexpr (kDataBits == 8) {
        block_attn::write_cache_v<cuteType, kDataBits>(
            tOrVt(_, j, i).data(),
            tOrVt(_, j + 1, i).data(),
            value_c8.data.elt + j,
            scale_v.data.elt + j,
            tidx,
            i);
      } else {
        block_attn::write_cache_v<cuteType, kDataBits>(
            tOrVt(_, j, i).data(),
            tOrVt(_, j + 1, i).data(),
            value_c4.data.elt + j / 2,
            scale_v.data.elt + j,
            tidx,
            i);
      }
    }
    if constexpr (kDataBits == 8) {
      value_c8.store_to(kv + i * kNThreads * 4 + tidx * 4);
    } else {
      value_c4.store_to(kv + i * kNThreads * 2 + tidx * 2);
    }
  }

  __syncthreads();
  pack_4_uint_vec* cache_v_cur =
      reinterpret_cast<pack_4_uint_vec*>(cache_v + cache_offset);

#pragma unroll
  for (int i = tidx; i < cache_data_num / pack_size; i += kNThreads) {
    cache_v_cur[i] = smem_pack_4[i];
  }
}

template <typename T>
class WriteEncoderCacheKvHdim128 {
 public:
  void operator()(const T* qkv_out,
                  const int* cu_seqlens_q,
                  const int* block_tables,
                  const int* seq_lens,
                  uint8_t* cache_k,
                  uint8_t* cache_v,
                  const T* cache_k_quant_scale,
                  const T* cache_k_zp,
                  const T* cache_v_quant_scale,
                  const T* cache_v_zp,
                  const std::string cache_quant_type_str,
                  const int num_head,
                  const int num_kv_head,
                  const int head_dim,
                  const int cache_k_group_num,
                  const int max_blocks_per_seq,
                  const int max_seq_len,
                  const int bsz,
                  cudaStream_t stream);
};

template <typename T>
void WriteEncoderCacheKvHdim128<T>::operator()(
    const T* qkv_out,
    const int* cu_seqlens_q,
    const int* block_tables,
    const int* seq_lens,
    uint8_t* cache_k,
    uint8_t* cache_v,
    const T* cache_k_quant_scale,
    const T* cache_v_quant_scale,
    const std::string cache_quant_type_str,
    const int num_head,
    const int num_kv_head,
    const int head_dim,
    const int cache_k_group_num,
    const int max_blocks_per_seq,
    const int max_seq_len,
    const int bsz,
    cudaStream_t stream) {
  assert(head_dim == 128);
  constexpr int kNThreads = 128;

  using cache_traits = CacheKV_quant_traits<T, 8>;
  dim3 grid_dim(max_blocks_per_seq, bsz, num_kv_head);
  int smem_size = cache_traits::kShareMemSize;
  auto kernel = &write_encoder_cache_kv_kernel<cache_traits, T, kNThreads>;

  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }

  kernel<<<grid_dim, kNThreads, smem_size, stream>>>(
      qkv_out,
      cu_seqlens_q,
      block_tables,
      seq_lens,
      cache_k,
      cache_v,
      const_cast<T*>(cache_k_quant_scale),
      const_cast<T*>(cache_v_quant_scale),
      num_head,
      num_kv_head,
      head_dim,
      max_blocks_per_seq);
}

template class WriteEncoderCacheKvHdim128<typename T>;
template class WriteEncoderCacheKvHdim128<phi::dtype::float16>;
#ifdef CUDA_BFLOAT16_AVALIABLE
template class WriteEncoderCacheKvHdim128<phi::dtype::bfloat16>;
#endif

template <typename CacheKV_traits, typename T, const int kNThreads>
inline __device__ void prefill_cache_kv(uint8_t* cache_k,
                                        uint8_t* cache_v,
                                        const int head_idx,
                                        const int row,
                                        const int col,
                                        const int tidx) {
  using cuteType = typename CacheKV_traits::cuteType;
  constexpr int pack_size = 16 / sizeof(cuteType);
  constexpr int kHeadDim = CacheKV_traits::kHeadDim;
  constexpr int kBlockSize = CacheKV_traits::kBlockSize;
  constexpr int kDataBits = CacheKV_traits::kDataBits;
  constexpr int kHeadDimKV = kHeadDim / (16 / kDataBits);
  __shared__ uint32_t kv[kBlockSize * kHeadDimKV / 2];
  using T_vec_4 = typename block_attn::Vec<cuteType, 4>;
  using T_vec_32 = typename block_attn::Vec<cuteType, 32>;
  using pack_8_T_vec = block_attn::Vec<cuteType, pack_size>;
  T_vec_4 value(static_cast<cuteType>(0.0f));
  T_vec_4 scale(static_cast<cuteType>(0.0f));
  pack_8_T_vec* smem_pack_8 = reinterpret_cast<pack_8_T_vec*>(kv);

  if (blockIdx.z == 0) {
#pragma unroll
    for (int i = 0; i < kHeadDim / 16; i++) {
      block_attn::write_cache_k<cuteType, kDataBits, kNThreads>(
          value.data.elt, value.data.elt, kv, scale.data.elt, tidx, i);
    }
    __syncthreads();
    pack_8_T_vec* cache_k_cur = reinterpret_cast<pack_8_T_vec*>(cache_k);

#pragma unroll
    for (int i = tidx; i < kBlockSize * kHeadDimKV / pack_size;
         i += kNThreads) {
      cache_k_cur[i] = smem_pack_8[i];
    }
  } else {
    using uint_vec_4 = typename block_attn::Vec<uint32_t, 4>;
    using uint_vec_2 = typename block_attn::Vec<uint32_t, 2>;
    uint_vec_4 value_c8;
    uint_vec_2 value_c4;
#pragma unroll
    for (int i = 0; i < kBlockSize / 16; i++) {
#pragma unroll
      for (int j = 0; j < kHeadDim / 32; j += 2) {
        if constexpr (kDataBits == 8) {
          block_attn::write_cache_v<cuteType, kDataBits>(value.data.elt,
                                                         value.data.elt,
                                                         value_c8.data.elt + j,
                                                         scale.data.elt,
                                                         tidx,
                                                         i);
        } else {
          block_attn::write_cache_v<cuteType, kDataBits>(
              value.data.elt,
              value.data.elt,
              value_c4.data.elt + j / 2,
              scale.data.elt,
              tidx,
              i);
        }
      }
      if constexpr (kDataBits == 8) {
        value_c8.store_to(kv + i * kNThreads * 4 + tidx * 4);
      } else {
        value_c4.store_to(kv + i * kNThreads * 2 + tidx * 2);
      }
    }

    __syncthreads();
    pack_8_T_vec* cache_v_cur = reinterpret_cast<pack_8_T_vec*>(cache_v);

#pragma unroll
    for (int i = tidx; i < kBlockSize * kHeadDimKV / pack_size;
         i += kNThreads) {
      cache_v_cur[i] = smem_pack_8[i];
    }
  }
}

template <typename CacheKV_traits, typename T, const int kNThreads>
__global__ void writeDecoderCacheKVKernel(T* qkv_input,
                                          T* qkv_bias,
                                          const float* rotary_emb,
                                          const int* block_tables,
                                          const int* seq_lens,
                                          uint8_t* cache_k,
                                          uint8_t* cache_v,
                                          T* cache_k_quant_scale,
                                          T* cache_v_quant_scale,
                                          const int head_num,
                                          const int kv_head_num,
                                          const int max_num_blocks_per_seq) {
  using index_t = typename CacheKV_traits::cache_index;
  using cuteType = typename CacheKV_traits::cuteType;
  constexpr int kHeadDim = CacheKV_traits::kHeadDim;
  constexpr int kBlockSize = CacheKV_traits::kBlockSize;
  constexpr int kDataBits = CacheKV_traits::kDataBits;
  constexpr int PackSize = kHeadDim / 32;
  using T_vec = block_attn::Vec<cuteType, PackSize>;
  using uint_vec = block_attn::Vec<uint32_t, PackSize>;
  constexpr int32_t cache_headdim = kDataBits == 4 ? kHeadDim / 2 : kHeadDim;
  const int bi = blockIdx.x;
  const int tidx = threadIdx.x;
  const int head_idx = blockIdx.y;
  const int kv_idx = blockIdx.z;
  const int seq_len = seq_lens[bi];
  if (seq_len == 0) {
    return;
  }
  const int col = tidx % 4;
  const int row = tidx / 4;
  const int warp_id = tidx / 32;
  const int lane_id = tidx % 32;

  using T_vec_4 = typename block_attn::Vec<cuteType, 4>;
  T_vec_4 scale_k = *reinterpret_cast<T_vec_4*>(
      cache_k_quant_scale + head_idx * kHeadDim + col * 32 + row * 4);

  T_vec_4 scale_v = *reinterpret_cast<T_vec_4*>(
      cache_v_quant_scale + head_idx * kHeadDim + lane_id * 4);

  constexpr int cache_k_cols = 4;
  __shared__ uint32_t cache_k_smem[8 * cache_k_cols];
  __shared__ cuteType data_k_smem[kHeadDim];
  __shared__ cuteType data_v_smem[kHeadDim];

  const int* block_table = block_tables + bi * max_num_blocks_per_seq;
  const int32_t block_idx = seq_len / kBlockSize;
  const int32_t block_offset = seq_len % kBlockSize;
  const index_t physical_block_number = block_table[block_idx];

  const int kv_load_idx =
      bi * (head_num + 2 * kv_head_num) * kHeadDim +
      (head_num + head_idx + kv_idx * kv_head_num) * kHeadDim;

  const index_t cache_offset =
      physical_block_number * kv_head_num * kBlockSize * cache_headdim +
      head_idx * kBlockSize * cache_headdim;

  if (block_offset == 0) {
    prefill_cache_kv<CacheKV_traits, T, kNThreads>(cache_k + cache_offset,
                                                   cache_v + cache_offset,
                                                   head_idx,
                                                   row,
                                                   col,
                                                   tidx);
    __syncthreads();
  }

  if (kv_idx == 0) {
    const int offset = block_offset / 32;
    int mask_cache = offset == 0 ? 0x0f0f0f0f : 0xf0f0f0f0;
    int mask_value = 0x0f0f0f0f;
    uint32_t* cache_k_cur = reinterpret_cast<uint32_t*>(cache_k + cache_offset);
    const int copy_idx = (block_offset * 4) % kHeadDim;

    if constexpr (kDataBits == 4) {
      if (tidx < 8) {
        *reinterpret_cast<uint_vec*>(cache_k_smem + tidx * 4) =
            *reinterpret_cast<uint_vec*>(cache_k_cur + tidx * kHeadDim +
                                         copy_idx);
      }
    } else {
      if (tidx < 8) {
        *reinterpret_cast<uint_vec*>(cache_k_smem + tidx * 4) =
            *reinterpret_cast<uint_vec*>(cache_k_cur + tidx * kHeadDim * 2 +
                                         copy_idx * 2 + offset * 4);
      }
    }

    if (warp_id == 1) {
      T_vec kv_src;
      T_vec kv_bias;
      kv_src.load_from(qkv_input + kv_load_idx + lane_id * PackSize);
      kv_bias.load_from(qkv_bias + (head_idx + head_num) * kHeadDim +
                        lane_id * PackSize);
      kv_src.add(kv_bias);

      block_attn::apply_rotary_embedding<cuteType, PackSize, kHeadDim>(
          kv_src, lane_id, rotary_emb + bi * kHeadDim);
      const int col_id = (col % 2) * 8 + col / 2 * 2 + (row - 8) * 16;
      *reinterpret_cast<uint32_t*>(data_k_smem + col_id) =
          *reinterpret_cast<uint32_t*>(kv_src.data.elt);
      *reinterpret_cast<uint32_t*>(data_k_smem + col_id + 4) =
          *reinterpret_cast<uint32_t*>(kv_src.data.elt + 2);
    }
    __syncthreads();
    if (warp_id == 0) {
      if constexpr (kDataBits == 4) {
        uint32_t value = block_attn::convert_half_2_c4_v2<cuteType, true, 4>(
            data_k_smem + lane_id * PackSize, scale_k.data.elt);
        value = (value & mask_value) << (4 - 4 * offset);
        uint32_t cache_value =
            cache_k_smem[row * cache_k_cols + col] & mask_cache;
        cache_value |= value;
        cache_k_smem[row * cache_k_cols + col] = cache_value;
        if (tidx < 8) {
          *reinterpret_cast<uint_vec*>(cache_k_cur + tidx * kHeadDim +
                                       copy_idx) =
              *reinterpret_cast<uint_vec*>(cache_k_smem + tidx * 4);
        }
      } else {
        uint32_t value = block_attn::convert_half_2_c8<cuteType, true, 4>(
            data_k_smem + lane_id * PackSize, scale_k.data.elt);
        cache_k_cur[row * kHeadDim * 2 + col * 2 + copy_idx * 2 + offset] =
            value;
      }
    }
  } else {
    if (warp_id == 0) {
      T_vec kv_src;
      T_vec kv_bias;
      kv_src.load_from(qkv_input + kv_load_idx + lane_id * PackSize);
      kv_bias.load_from(qkv_bias +
                        (head_idx + head_num + kv_head_num) * kHeadDim +
                        lane_id * PackSize);
      kv_src.add(kv_bias);
      kv_src.store_to(data_v_smem + lane_id * PackSize);
    }
    __syncthreads();
    uint32_t* cache_v_cur = reinterpret_cast<uint32_t*>(cache_v + cache_offset);

    if (warp_id == 0) {
      const int copy_idx = (block_offset % 16) / 8 * 2 + block_offset % 2;
      const int hid_idx = block_offset / 16;

      T_vec value_vec;
      for (int j = 0; j < PackSize; ++j) {
        value_vec.data.elt[j] = data_v_smem[lane_id + j * 32];
      }

      if constexpr (kDataBits == 4) {
        uint32_t value = block_attn::convert_half_2_c4_v2<cuteType, true, 4>(
            value_vec.data.elt, scale_v.data.elt);
        using uint_vec_2 = block_attn::Vec<uint32_t, 2>;
        uint_vec_2 cache_v_cur_2;
        const uint32_t mask_value = 0xff000000 >> (8 * copy_idx);
        const uint32_t mask_cache = ~mask_value;
        const int cache_cur_idx = lane_id * PackSize * 2 +
                                  (block_offset % 8) / 2 * 2 +
                                  hid_idx * kHeadDim * 2;
        cache_v_cur_2.load_from(cache_v_cur + cache_cur_idx);
        value = (value >> 4) | value;
        uint32_t value1;
        if (copy_idx > 1) {
          value1 = (value & 0x00ff0000) >> (copy_idx - 1) * 8;
        } else {
          value1 = (value & 0x00ff0000) << (1 - copy_idx) * 8;
        }
        uint32_t value2 = (value & 0x000000ff) << (3 - copy_idx) * 8;
        cache_v_cur_2.data.elt[0] =
            (cache_v_cur_2.data.elt[0] & mask_cache) | value1;
        cache_v_cur_2.data.elt[1] =
            (cache_v_cur_2.data.elt[1] & mask_cache) | value2;
        cache_v_cur_2.store_to(cache_v_cur + cache_cur_idx);
      } else {
        uint32_t value = block_attn::convert_half_2_c8<cuteType, true, 4>(
            value_vec.data.elt, scale_v.data.elt);
        using uint_vec_4 = block_attn::Vec<uint32_t, 4>;
        uint_vec_4 cache_v_cur_4;
        const int cache_cur_idx = lane_id * PackSize * 4 +
                                  (block_offset % 8) / 2 * 4 +
                                  hid_idx * kHeadDim * 4;
        cache_v_cur_4.load_from(cache_v_cur + cache_cur_idx);
#pragma unroll
        for (int j = 0; j < PackSize; ++j) {
          reinterpret_cast<uint8_t*>(cache_v_cur_4.data.elt + j)[copy_idx] =
              reinterpret_cast<uint8_t*>(&value)[j];
        }
        cache_v_cur_4.store_to(cache_v_cur + cache_cur_idx);
      }
    }
  }
}

template <typename CacheKV_traits, typename T, const int kNThreads>
__global__ void writeC16DecoderCacheKVKernel(T* qkv_input,
                                             T* qkv_bias,
                                             const float* rotary_emb,
                                             const int* block_tables,
                                             const int* seq_lens,
                                             T* cache_k,
                                             T* cache_v,
                                             const int head_num,
                                             const int kv_head_num,
                                             const int max_num_blocks_per_seq) {
  using index_t = typename CacheKV_traits::cache_index;
  using cuteType = typename CacheKV_traits::cuteType;
  constexpr int kHeadDim = CacheKV_traits::kHeadDim;
  constexpr int kBlockSize = CacheKV_traits::kBlockSize;
  constexpr int PackSize = kHeadDim / 32;
  static_assert(kHeadDim == 128);
  using T_vec = block_attn::Vec<cuteType, PackSize>;
  const int bi = blockIdx.x;
  const int tidx = threadIdx.x;
  const int head_idx = blockIdx.y;
  const int seq_len = seq_lens[bi];
  if (seq_len == 0) {
    return;
  }
  const int warp_id = tidx / 32;
  const int lane_id = tidx % 32;

  const int* block_table = block_tables + bi * max_num_blocks_per_seq;
  const int32_t block_idx = seq_len / kBlockSize;
  const int32_t block_offset = seq_len % kBlockSize;
  const index_t physical_block_number = block_table[block_idx];

  const int kv_bias_idx =
      (head_num + head_idx + warp_id * kv_head_num) * kHeadDim +
      lane_id * PackSize;
  const int kv_load_idx =
      bi * (head_num + 2 * kv_head_num) * kHeadDim + kv_bias_idx;

  const index_t cache_offset =
      physical_block_number * kv_head_num * kBlockSize * kHeadDim +
      head_idx * kBlockSize * kHeadDim + block_offset * kHeadDim +
      lane_id * PackSize;

  T_vec kv_src;
  T_vec kv_bias;
  kv_src.load_from(qkv_input + kv_load_idx);

  kv_bias.load_from(qkv_bias + kv_bias_idx);
  kv_src.add(kv_bias);

  if (warp_id == 0) {
    block_attn::apply_rotary_embedding<cuteType, PackSize, kHeadDim>(
        kv_src, lane_id, rotary_emb + bi * kHeadDim);
    kv_src.store_to(cache_k + cache_offset);
  } else {
    kv_src.store_to(cache_v + cache_offset);
  }
}

template <typename T, int cachenbits>
class WriteDecoderCacheKvHdim128 {
 public:
  void operator()(const T* qkv_input,
                  const T* qkv_bias,
                  const float* rotary_emb,
                  const int* block_tables,
                  const int* seq_lens,
                  uint8_t* cache_k,
                  uint8_t* cache_v,
                  const T* cache_k_quant_scale,
                  const T* cache_k_zp,
                  const T* cache_v_quant_scale,
                  const T* cache_v_zp,
                  const int bsz,
                  const int num_head,
                  const int kv_head_num,
                  const int head_dim,
                  const int max_num_blocks_per_seq,
                  cudaStream_t stream);
};

template <typename T, int cachenbits>
void WriteDecoderCacheKvHdim128<T, cachenbits>::operator()(
    const T* qkv_input,
    const T* qkv_bias,
    const float* rotary_emb,
    const int* block_tables,
    const int* seq_lens,
    uint8_t* cache_k,
    uint8_t* cache_v,
    const T* cache_k_quant_scale,
    const T* cache_v_quant_scale,
    const int bsz,
    const int num_head,
    const int kv_head_num,
    const int head_dim,
    const int max_num_blocks_per_seq,
    cudaStream_t stream) {
  constexpr int ThreadsPerBlock = 128;
  assert(head_dim == 128);
  using cache_traits = CacheKV_quant_traits<T, cachenbits>;
  dim3 grid_dim;
  grid_dim.x = bsz;
  grid_dim.y = kv_head_num;
  grid_dim.z = 2;
  auto kernel = &writeDecoderCacheKVKernel<cache_traits, T, ThreadsPerBlock>;

  kernel<<<grid_dim, ThreadsPerBlock, 0, stream>>>(
      const_cast<T*>(qkv_input),
      const_cast<T*>(qkv_bias),
      rotary_emb,
      block_tables,
      seq_lens,
      cache_k,
      cache_v,
      const_cast<T*>(cache_k_quant_scale),
      const_cast<T*>(cache_v_quant_scale),
      num_head,
      kv_head_num,
      max_num_blocks_per_seq);
}

template <typename T>
class WriteC16DecoderCacheKvHdim128 {
 public:
  void operator()(const T* qkv_input,
                  const T* qkv_bias,
                  const float* rotary_emb,
                  const int* block_tables,
                  const int* seq_lens,
                  T* cache_k,
                  T* cache_v,
                  const int bsz,
                  const int num_head,
                  const int kv_head_num,
                  const int head_dim,
                  const int max_num_blocks_per_seq,
                  cudaStream_t stream);
};

template <typename T>
void WriteC16DecoderCacheKvHdim128<T>::operator()(
    const T* qkv_input,
    const T* qkv_bias,
    const float* rotary_emb,
    const int* block_tables,
    const int* seq_lens,
    T* cache_k,
    T* cache_v,
    const int bsz,
    const int num_head,
    const int kv_head_num,
    const int head_dim,
    const int max_num_blocks_per_seq,
    cudaStream_t stream) {
  constexpr int ThreadsPerBlock = 64;
  assert(head_dim == 128);
  using cache_traits = CacheKV_quant_traits<T, 16>;
  dim3 grid_dim;
  grid_dim.x = bsz;
  grid_dim.y = kv_head_num;
  auto kernel = &writeC16DecoderCacheKVKernel<cache_traits, T, ThreadsPerBlock>;

  kernel<<<grid_dim, ThreadsPerBlock, 0, stream>>>(const_cast<T*>(qkv_input),
                                                   const_cast<T*>(qkv_bias),
                                                   rotary_emb,
                                                   block_tables,
                                                   seq_lens,
                                                   cache_k,
                                                   cache_v,
                                                   num_head,
                                                   kv_head_num,
                                                   max_num_blocks_per_seq);
}

template class WriteDecoderCacheKvHdim128<phi::dtype::float16, 8>;
template class WriteDecoderCacheKvHdim128<phi::dtype::bfloat16, 8>;

template class WriteC16DecoderCacheKvHdim128<phi::dtype::float16>;
template class WriteC16DecoderCacheKvHdim128<phi::dtype::bfloat16>;

template <bool Is_first,
          int kMiLen,
          typename Tensor0,
          typename Tensor1,
          typename T>
inline __device__ void softmax_rescale_o(Tensor0& scores,
                                         Tensor1& acc_o,
                                         const T* scores_max,
                                         const T* scores_max_prev,
                                         T* scores_sum,
                                         const float softmax_scale) {
  if (Is_first) {
    block_attn::scale_apply_exp2<kMiLen>(
        scores, scores_max, scores_sum, softmax_scale);
  } else {
    Tensor acc_o_rowcol = make_tensor(
        acc_o.data(), block_attn::convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < kMiLen; ++mi) {
      const float scores_scale =
          expf((scores_max_prev[mi] - scores_max[mi]) * softmax_scale);
      scores_sum[mi] *= scores_scale;
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale;
      }
    }
    block_attn::scale_apply_exp2<kMiLen>(
        scores, scores_max, scores_sum, softmax_scale);
  }
};

template <typename Kernel_traits, typename ParamType>
__global__ __launch_bounds__(
    Kernel_traits::
        kNThreads) void multi_block_gqa_attention_kernel(ParamType params) {
  // qkv的shape为[token_num, head num + 2 * kv_head_num, head dim]
  // 其中decoder和encoder的结果在一起 qkv_bias的shape为[head num + 2 *
  // kv_head_num, head dim] attn_out的shape为[bsz, head num, head dim]
  // partion_out的shape为[bsz, head num, partition num, head dim]
  using cuteType = typename Kernel_traits::cuteType;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;
  using CacheKV_traits = typename Kernel_traits::CacheKV_traits;
  constexpr int32_t kHeadDim = Kernel_traits::kHeadDim;
  constexpr int32_t kHeadDimKV = Kernel_traits::kHeadDimKV;
  constexpr int32_t kBlockM = Kernel_traits::kBlockM;
  constexpr int32_t kBlockSize = Kernel_traits::kBlockSize;
  constexpr int32_t kGqaGroupSize = Kernel_traits::kGqaGroupSize;
  constexpr int32_t kNWarps = Kernel_traits::kNWarps;
  constexpr int32_t kTileN = Kernel_traits::kTileN;
  constexpr int32_t kBlockN = kTileN * kBlockSize;
  constexpr int32_t kDataBits = Kernel_traits::kDataBits;
  constexpr int32_t kMiLen = (kGqaGroupSize + 7) / 8;

  const int32_t bi = blockIdx.y;
  const int32_t tidx = threadIdx.x;
  const int32_t partition_idx = blockIdx.x;
  const int32_t kv_head_idx = blockIdx.z;
  const int32_t q_head_idx = kv_head_idx * kGqaGroupSize;
  // +1 是因为 decoder 的kv已经写入cache kv
  const int32_t seq_len = params.seq_lens[bi] + 1;

  const int32_t head_num = params.head_num;
  const int32_t kv_head_num = params.kv_head_num;

  const int32_t partition_num = (seq_len + kBlockN - 1) / kBlockN;

  if (seq_len == 0 || partition_idx >= partition_num) {
    return;
  }

  const int q_bias_offset = q_head_idx * kHeadDim;

  cuteType* q_input = reinterpret_cast<cuteType*>(params.qkv_input) +
                      bi * (head_num + 2 * kv_head_num) * kHeadDim;

  Tensor gQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<const cuteType*>(q_input) + q_bias_offset),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      Stride<Int<kHeadDim>, _1>{});

  Tensor gQbias = make_tensor(
      make_gmem_ptr(reinterpret_cast<const cuteType*>(params.qkv_bias) +
                    q_bias_offset),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      Stride<Int<kHeadDim>, _1>{});

  const int32_t block_idx = partition_idx * kTileN;
  const int* block_table =
      params.block_table + bi * params.max_num_blocks_per_seq + block_idx;
  const index_t physical_block_number = block_table[0];

  const index_t cache_offset =
      (physical_block_number * kv_head_num + kv_head_idx) * kBlockSize *
      kHeadDimKV;

  Tensor gK = make_tensor(
      make_gmem_ptr(reinterpret_cast<const cuteType*>(params.cache_k) +
                    cache_offset),
      Shape<Int<kBlockSize>, Int<kHeadDimKV>>{},
      Stride<Int<kHeadDimKV>, _1>{});

  Tensor gV = make_tensor(
      make_gmem_ptr(reinterpret_cast<const cuteType*>(params.cache_v) +
                    cache_offset),
      Shape<Int<kBlockSize>, Int<kHeadDimKV>>{},
      Stride<Int<kHeadDimKV>, _1>{});

  // Shared memory.
  extern __shared__ char smem_[];
  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<cuteType*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sQbias =
      make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutQ{});
  Tensor sQK = make_tensor(sQbias.data() + size(sQbias),
                           typename Kernel_traits::SmemLayoutQK{});

  Tensor sK = make_tensor(sQK.data() + size(sQK),
                          typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
  __shared__ ElementAccum scores_warp[kNWarps][kMiLen * kBlockM];

  auto gmem_thr_copy_Q =
      typename Kernel_traits::GmemTiledCopyQ{}.get_thread_slice(tidx);
  auto gmem_thr_copy_KV =
      typename Kernel_traits::GmemTiledCopyKV{}.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
  Tensor tQgQbias = gmem_thr_copy_Q.partition_S(gQbias);
  Tensor tQsQbias = gmem_thr_copy_Q.partition_D(sQbias);

  Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);

  Tensor cQ = make_identity_tensor(make_shape(kBlockM, kHeadDim));
  Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);

  Tensor cKV = make_identity_tensor(make_shape(kBlockSize, kHeadDim));
  Tensor tKVcKV = gmem_thr_copy_KV.partition_S(cKV);

  typename Kernel_traits::TiledMma tiled_mma;

  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  using SmemCopyAtom = typename Kernel_traits::SmemCopyAtom;
  auto smem_thr_copy_Q =
      make_tiled_copy_A(SmemCopyAtom{}, tiled_mma).get_thread_slice(tidx);
  auto smem_thr_copy_K =
      make_tiled_copy_B(SmemCopyAtom{}, tiled_mma).get_thread_slice(tidx);
  auto smem_thr_copy_V =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{},
                        tiled_mma)
          .get_thread_slice(tidx);
  auto smem_thr_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma)
          .get_thread_slice(tidx);

  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

  Tensor tSsQK = smem_thr_copy_Q.partition_S(sQK);
  Tensor tSrQK = thr_mma.partition_fragment_A(sQK);
  Tensor tSsQbias = smem_thr_copy_Q.partition_S(sQbias);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

  block_attn::copy<false>(gmem_thr_copy_Q, tQgQ, tQsQ, tQcQ, kGqaGroupSize);
  block_attn::copy<false>(
      gmem_thr_copy_Q, tQgQbias, tQsQbias, tQcQ, kGqaGroupSize);

  cute::cp_async_fence();
  block_attn::cp_async_wait<0>();

  const int32_t remain_seq_len = seq_len - partition_idx * kTileN * kBlockSize;

  block_attn::copy(gmem_thr_copy_KV, tKgK, tKsK, tKVcKV);

  cute::cp_async_fence();

  float* rotary_emb = params.rotary_emb + bi * kHeadDim;

  const int32_t warp_id = tidx / 32;
  const int32_t lane_id = tidx % 32;
  const int32_t row = lane_id / 4;
  const int32_t col = lane_id % 4;
  const int row_idx = tidx / 4;
  __syncthreads();
#pragma unroll
  for (int i = tidx * 2; i < kGqaGroupSize * kHeadDim;
       i += 2 * Kernel_traits::kNThreads) {
    const int row_index = i / kHeadDim;
    const int col_index = i % kHeadDim;
    float bias1 = static_cast<float>(sQbias(row_index, col_index));
    float bias2 = static_cast<float>(sQbias(row_index, col_index + 1));
    float v1 = static_cast<float>(sQ(row_index, col_index)) + bias1;
    float v2 = static_cast<float>(sQ(row_index, col_index + 1)) + bias2;
    float cos = rotary_emb[col_index];
    float sin = rotary_emb[col_index + 1];
    sQ(row_index, col_index) = static_cast<cuteType>(cos * v1 - sin * v2);
    sQ(row_index, col_index + 1) = static_cast<cuteType>(sin * v1 + cos * v2);
  }

  using T_vec_4 = typename block_attn::Vec<cuteType, 4>;
  using T_vec_32 = typename block_attn::Vec<cuteType, 32>;

  const T_vec_32 scale_k = *reinterpret_cast<const T_vec_32*>(
      params.cache_k_dequant_scale + kv_head_idx * kHeadDim + col * 32);
  const T_vec_4 scale_v = *reinterpret_cast<const T_vec_4*>(
      params.cache_v_dequant_scale + kv_head_idx * kHeadDim + row_idx * 4);

  Tensor acc_o =
      partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);
  Tensor acc_s =
      partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockSize>>{});

  ElementAccum scores_max[kMiLen];
  ElementAccum scores_max_prev[kMiLen];
  ElementAccum scores_sum[kMiLen];

#pragma unroll
  for (int mi = 0; mi < kMiLen; ++mi) {
    scores_max[mi] = -INFINITY;
    scores_sum[mi] = 0;
  }

  const int cache_offset_step = kv_head_num * kBlockSize * kHeadDimKV;

#pragma unroll
  for (int n = 0; n < kTileN; ++n) {
    const int cur_remain_seq_len = remain_seq_len - n * kBlockSize;

    if (cur_remain_seq_len <= 0) {
      break;
    }

    clear(acc_s);
    block_attn::cp_async_wait<0>();
    __syncthreads();

    if (n > 0) {
      tVgV.data() = tVgV.data() +
                    (block_table[n] - block_table[n - 1]) * cache_offset_step;
    }

    block_attn::copy(gmem_thr_copy_KV, tVgV, tVsV, tKVcKV);

    cute::cp_async_fence();

    if constexpr (kDataBits == 16) {
      if (n == 0) {
        block_attn::gemm(acc_s,
                         tSrQ,
                         tSrK,
                         tSsQ,
                         tSsK,
                         tiled_mma,
                         smem_thr_copy_Q,
                         smem_thr_copy_K);
      } else {
        block_attn::gemm<true>(acc_s,
                               tSrQ,
                               tSrK,
                               tSsQ,
                               tSsK,
                               tiled_mma,
                               smem_thr_copy_Q,
                               smem_thr_copy_K);
      }
    } else {
      Tensor tSrKQuant = make_tensor<cuteType>(
          Layout<Shape<Shape<_2, _2>, Int<kBlockSize / 32>>,
                 Stride<Shape<_1, _2>, _4>>{});
      if (n == 0) {
        block_attn::
            gemm_qk_quant<CacheKV_traits, cuteType, kHeadDim, kDataBits>(
                acc_s,
                tSrQ,
                tSsQ,
                tSrKQuant,
                sK,
                tiled_mma,
                smem_thr_copy_Q,
                tidx,
                scale_k.data.elt);
      } else {
        block_attn::
            gemm_qk_quant<CacheKV_traits, cuteType, kHeadDim, kDataBits, true>(
                acc_s,
                tSrQ,
                tSsQ,
                tSrKQuant,
                sK,
                tiled_mma,
                smem_thr_copy_Q,
                tidx,
                scale_k.data.elt);
      }
    }
    Tensor scores = make_tensor(
        acc_s.data(), block_attn::convert_layout_acc_rowcol(acc_s.layout()));

    if (partition_idx == partition_num - 1 && cur_remain_seq_len < kBlockSize) {
      block_attn::apply_mask<kMiLen>(scores, warp_id, col, cur_remain_seq_len);
    }

#pragma unroll
    for (int mi = 0; mi < kMiLen; ++mi) {
      scores_max_prev[mi] = scores_max[mi];
    }

    block_attn::reduce_max<kMiLen>(scores, scores_max);

    if (col == 0) {
      scores_warp[warp_id][row] = scores_max[0];
      if constexpr (kMiLen > 1) {
        scores_warp[warp_id][row + 8] = scores_max[1];
      }
    }

    __syncthreads();

    block_attn::MaxOp<ElementAccum> max_op;

    if (tidx < kGqaGroupSize) {
      float cur_max = scores_warp[0][tidx];
#pragma unroll
      for (uint32_t i = 1; i < kNWarps; ++i) {
        cur_max = max_op(scores_warp[i][tidx], cur_max);
      }
      scores_warp[0][tidx] = cur_max;
    }

    block_attn::cp_async_wait<0>();
    __syncthreads();

    if (cur_remain_seq_len > kBlockSize && n < kTileN - 1) {
      tKgK.data() = tKgK.data() +
                    (block_table[n + 1] - block_table[n]) * cache_offset_step;
      block_attn::copy(gmem_thr_copy_KV, tKgK, tKsK, tKVcKV);
      cute::cp_async_fence();
    }

#pragma unroll
    for (int mi = 0; mi < kMiLen; ++mi) {
      scores_max[mi] = scores_warp[0][row + mi * 8];
    }

    if (n == 0) {
      softmax_rescale_o<true, kMiLen>(scores,
                                      acc_o,
                                      scores_max,
                                      scores_max_prev,
                                      scores_sum,
                                      params.inv_sqrt_dh);
    } else {
      softmax_rescale_o<false, kMiLen>(scores,
                                       acc_o,
                                       scores_max,
                                       scores_max_prev,
                                       scores_sum,
                                       params.inv_sqrt_dh);
    }

    Tensor rS = block_attn::convert_type<cuteType>(acc_s);

    Tensor trQK = smem_thr_copy_O.retile_S(rS);
    Tensor tsQK = smem_thr_copy_O.partition_D(sQK);
    cute::copy(smem_thr_copy_O, trQK, tsQK);

    __syncthreads();

    if constexpr (kDataBits == 16) {
      block_attn::gemm(acc_o,
                       tSrQK,
                       tOrVt,
                       tSsQK,
                       tOsVt,
                       tiled_mma,
                       smem_thr_copy_Q,
                       smem_thr_copy_V);
    } else {
      Tensor tSrVQuant = make_tensor<cuteType>(
          Layout<Shape<_4, Shape<_2, _2>>, Stride<_1, Shape<_4, _8>>>{});
      block_attn::
          gemm_value_quant<CacheKV_traits, cuteType, kHeadDim, kDataBits>(
              acc_o,
              tSrQK,
              tSsQK,
              tSrVQuant,
              sV,
              tiled_mma,
              smem_thr_copy_Q,
              tidx,
              scale_v.data.elt);
    }
  }

  const uint32_t pack_max_partition_num =
      (params.max_num_partitions + 3) / 4 * 4;
  uint32_t max_sum_offset = bi * pack_max_partition_num * head_num +
                            (tidx + q_head_idx) * pack_max_partition_num +
                            partition_idx;

  if (tidx < kGqaGroupSize) {
    params.maxs[max_sum_offset] = scores_warp[0][tidx] * params.inv_sqrt_dh;
  }

  block_attn::SumOp<ElementAccum> sum_op;
#pragma unroll
  for (int mi = 0; mi < kMiLen; ++mi) {
    scores_sum[mi] = block_attn::Allreduce<4>::run(scores_sum[mi], sum_op);
  }
  __syncthreads();

  if (col == 0) {
    scores_warp[warp_id][row] = scores_sum[0];
    if constexpr (kMiLen > 1) {
      scores_warp[warp_id][row + 8] = scores_sum[1];
    }
  }

  Tensor rO = block_attn::convert_type<cuteType>(acc_o);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sQ);

  cute::copy(smem_thr_copy_O, taccOrO, taccOsO);

  __syncthreads();

  if (tidx < kGqaGroupSize) {
    float cur_sum = scores_warp[0][tidx];
#pragma unroll
    for (uint32_t i = 1; i < kNWarps; ++i) {
      cur_sum = sum_op(scores_warp[i][tidx], cur_sum);
    }
    scores_warp[0][tidx] = cur_sum;
  }

  Tensor gO = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<cuteType*>(params.partition_attn_out) +
          ((bi * params.max_num_partitions + partition_idx) * head_num +
           q_head_idx) *
              kHeadDim),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      Stride<Int<kHeadDim>, _1>{});

  auto gmem_thr_copy_O =
      typename Kernel_traits::GmemTiledCopyO{}.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sQ);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
  constexpr int32_t copy_size = kGqaGroupSize * 16;
  __syncthreads();

  if (tidx < copy_size) {
    cute::copy(gmem_thr_copy_O, tOsO(_, 0, _), tOgO(_, 0, _));
  }

  if constexpr (kMiLen > 1) {
    if (tidx < copy_size - 128) {
      cute::copy(gmem_thr_copy_O, tOsO(_, 1, _), tOgO(_, 1, _));
    }
  }
  if (tidx < kGqaGroupSize) {
    params.sums[max_sum_offset] = scores_warp[0][tidx];
  }
}

template <typename Kernel_traits, typename ParamType>
inline __device__ float caluate_logit_scale(const int partition_num,
                                            const int pack_max_partition_num,
                                            ParamType& params,
                                            char* shared_mem) {
  constexpr int32_t kNFloatPacksize = 16 / sizeof(float);
  constexpr int32_t kNReduceThreads = Kernel_traits::kNReduceThreads;
  const int32_t bi = blockIdx.z;
  const int32_t tidx = threadIdx.x;
  const int32_t head_idx = blockIdx.y;
  const int32_t head_num = params.head_num;

  using float_vec = block_attn::Vec<float, kNFloatPacksize>;
  const int32_t offset = bi * head_num * pack_max_partition_num +
                         head_idx * pack_max_partition_num;

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = params.maxs + offset;
  float global_max_logit = -FLT_MAX;

  int32_t idx = tidx * kNFloatPacksize;
#pragma unroll
  for (; idx <= partition_num - kNFloatPacksize;
       idx += kNReduceThreads * kNFloatPacksize) {
    float_vec cur_max =
        *reinterpret_cast<const float_vec*>(max_logits_ptr + idx);
#pragma unroll
    for (int32_t j = 0; j < kNFloatPacksize; ++j) {
      global_max_logit = fmaxf(global_max_logit, cur_max.data.elt[j]);
    }
    cur_max.store_to(shared_max_logits + idx);
  }

  const int32_t packed_data_num =
      partition_num / kNFloatPacksize * kNFloatPacksize;
  // 剩下的不足packsize的元素
  idx = packed_data_num + tidx;
#pragma unroll
  for (; idx < partition_num; idx += kNReduceThreads) {
    float cur_max = max_logits_ptr[idx];
    global_max_logit = fmaxf(global_max_logit, cur_max);
    shared_max_logits[idx] = cur_max;
  }
  __syncthreads();
  // Reduce within the warp.
  global_max_logit =
      block_attn::BlockAllReduce<float,
                                 block_attn::MaxOp<float>,
                                 kNReduceThreads>(global_max_logit);
  // Load rescaled exp sums to shared memory.
  float* share_sum_scale = reinterpret_cast<float*>(
      shared_mem + sizeof(float) * pack_max_partition_num);
  const float* exp_sums_ptr = params.sums + offset;
  float global_exp_sum = 0.0f;

  idx = tidx * kNFloatPacksize;
#pragma unroll
  for (; idx <= partition_num - kNFloatPacksize;
       idx += kNReduceThreads * kNFloatPacksize) {
    float_vec share_max =
        *reinterpret_cast<const float_vec*>(shared_max_logits + idx);
#pragma unroll
    for (int32_t j = 0; j < kNFloatPacksize; ++j) {
      float exp_sub_max = expf(share_max.data.elt[j] - global_max_logit);
      float rescaled_exp_sum = exp_sums_ptr[idx + j] * exp_sub_max;
      global_exp_sum += rescaled_exp_sum;
      share_max.data.elt[j] = exp_sub_max;
    }
    share_max.store_to(share_sum_scale + idx);
  }

  // 剩下的不足packsize的元素
  idx = packed_data_num + tidx;
#pragma unroll
  for (; idx < partition_num; idx += kNReduceThreads) {
    float share_max = shared_max_logits[idx];
    float exp_sub_max = expf(share_max - global_max_logit);
    float rescaled_exp_sum = exp_sums_ptr[idx] * exp_sub_max;
    global_exp_sum += rescaled_exp_sum;
    share_sum_scale[idx] = exp_sub_max;
  }
  __syncthreads();

  global_exp_sum = block_attn::BlockAllReduce<float,
                                              block_attn::SumOp<float>,
                                              kNReduceThreads>(global_exp_sum);

  const float inv_global_exp_sum = fdividef(1.0f, global_exp_sum + 1e-6f);
  return inv_global_exp_sum;
}

template <typename Kernel_traits, typename ParamType>
__global__ void __launch_bounds__(Kernel_traits::kNReduceThreads)
    multi_block_attention_reduce_kernel(ParamType params) {
  using cuteType = typename Kernel_traits::cuteType;
  constexpr int32_t kBlockN = Kernel_traits::kTileN * Kernel_traits::kBlockSize;
  constexpr int32_t kNReducePacksize = 16 / sizeof(cuteType);
  constexpr int32_t kNFloatPacksize = 16 / sizeof(float);
  constexpr int32_t kNReduceWarps = Kernel_traits::kNReduceWarps;
  constexpr int32_t kHeadDim = Kernel_traits::kHeadDim;
  const int32_t bi = blockIdx.z;
  const int32_t headdim_idx = kNReducePacksize * kNReduceWarps * blockIdx.x;
  const int32_t tidx = threadIdx.x;
  const int32_t head_idx = blockIdx.y;
  const int32_t warp_id = tidx / 32;
  const int32_t lane_id = tidx % 32;
  const int32_t seq_len = params.seq_lens[bi];
  const int32_t head_num = params.head_num;
  using pack_half = typename block_attn::PackedHalf<cuteType>::Type;

  if (seq_len == 0) {
    return;
  }

  extern __shared__ char shared_mem[];

  const int32_t partition_num = (seq_len + kBlockN) / kBlockN;
  const int32_t pack_max_partition_num =
      (params.max_num_partitions + kNFloatPacksize - 1) / kNFloatPacksize *
      kNFloatPacksize;

  float* share_sum_scale = reinterpret_cast<float*>(
      shared_mem + sizeof(float) * pack_max_partition_num);

  float inv_global_exp_sum = caluate_logit_scale<Kernel_traits>(
      partition_num, pack_max_partition_num, params, shared_mem);

  using T_vec = block_attn::Vec<cuteType, kNReducePacksize>;

  cuteType* partition_attn_out =
      reinterpret_cast<cuteType*>(params.partition_attn_out) +
      bi * head_num * params.max_num_partitions * kHeadDim +
      head_idx * kHeadDim + headdim_idx;

  block_attn::Vec<float, kNReducePacksize> acc(0.0f);
#pragma unroll
  for (int idx = lane_id; idx < partition_num; idx += 32) {
    T_vec sub_logits = *reinterpret_cast<T_vec*>(
        &partition_attn_out[idx * head_num * kHeadDim +
                            warp_id * kNReducePacksize]);
    float scale = share_sum_scale[idx];
#pragma unroll
    for (int k = 0; k < kNReducePacksize; ++k) {
      acc.data.elt[k] += static_cast<float>(sub_logits.data.elt[k]) * scale;
    }
  }

  __syncthreads();
  using out_type = typename ParamType::out_type;
  T_vec out;
#pragma unroll
  for (int k = 0; k < kNReducePacksize; ++k) {
    out.data.elt[k] = static_cast<cuteType>(
        block_attn::WarpAllReduce<float, block_attn::SumOp<float>>(
            acc.data.elt[k]) *
        inv_global_exp_sum);
  }

  const int ori_token_idx =
      bi * params.max_input_length - params.cum_offsets[bi];
  const int smooth_offset =
      head_idx * kHeadDim + headdim_idx + warp_id * kNReducePacksize;
  out_type* attn_out = reinterpret_cast<out_type*>(params.attn_out) +
                       ori_token_idx * head_num * kHeadDim + smooth_offset;

  if (lane_id == 0) {
    out.store_to(attn_out);
  }
}

template <typename Kernel_traits, typename ParamType>
void run_block_attn(ParamType& params, cudaStream_t stream) {
  dim3 grid;
  // 最后一个block做当前seq，剩下的做cache kv的
  grid.x = params.max_num_partitions;
  grid.y = params.batch_size;
  grid.z = params.kv_head_num;
  constexpr int smem_size = Kernel_traits::kShareMemSize;
  constexpr auto kernel =
      &multi_block_gqa_attention_kernel<Kernel_traits, ParamType>;
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
  kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
  // 2 是指 存放max和sum
  int32_t reduce_shared_mem_size =
      2 * (params.max_num_partitions + 4) * sizeof(float);
  constexpr int32_t pack_size = 16 / sizeof(typename ParamType::value_type);
  static_assert(Kernel_traits::kHeadDim % pack_size == 0);
  static_assert((Kernel_traits::kHeadDim / Kernel_traits::kNReduceWarps) %
                    pack_size ==
                0);
  grid.x = Kernel_traits::kHeadDim / pack_size;
  grid.y = params.head_num;
  grid.z = params.batch_size;

  auto reduce_kernel =
      &multi_block_attention_reduce_kernel<Kernel_traits, ParamType>;
  grid.x = Kernel_traits::kHeadDim / Kernel_traits::kNReduceWarps / pack_size;

  if (reduce_shared_mem_size >= 48 * 1024) {
    cudaFuncSetAttribute(reduce_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         reduce_shared_mem_size);
  }
  reduce_kernel<<<grid,
                  Kernel_traits::kNReduceThreads,
                  reduce_shared_mem_size,
                  stream>>>(params);
}

template <typename T, typename CacheKVTraits, typename ParamType, int kBlockN>
void run_block_attn_hdim128(ParamType& params, cudaStream_t stream) {
  const int gqaGroupSize = params.head_num / params.kv_head_num;
  constexpr int kTileN = kBlockN / 64;
  switch (gqaGroupSize) {
    case 1: {
      run_block_attn<Block_attn_kernel_traits<1, kTileN, CacheKVTraits>>(
          params, stream);
      break;
    }
    case 4: {
      run_block_attn<Block_attn_kernel_traits<4, kTileN, CacheKVTraits>>(
          params, stream);
      break;
    }
    case 8: {
      run_block_attn<Block_attn_kernel_traits<8, kTileN, CacheKVTraits>>(
          params, stream);
      break;
    }
    case 12: {
      run_block_attn<Block_attn_kernel_traits<12, kTileN, CacheKVTraits>>(
          params, stream);
      break;
    }
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "DecoderBlockAttention not implemented for gqaGroupSize = %d",
          gqaGroupSize));
    }
  }
}

template <typename T,
          typename Context,
          typename out_type,
          int max_seq_per_block,
          int cachenbits>
void set_params(const phi::GPUContext& dev_ctx,
                const phi::DenseTensor& qkv_tensor,
                const phi::DenseTensor* qkv_bias_tensor,
                const phi::DenseTensor* block_tables,
                const phi::DenseTensor* cum_offsets_tensor,
                const phi::DenseTensor* sequence_lengths_tensor,
                const phi::DenseTensor* rotary_tensor,
                phi::DenseTensor* k_cache,
                phi::DenseTensor* v_cache,
                phi::DenseTensor* out_tensor,
                const int batch_size,
                const int max_num_blocks_per_seq,
                const int max_input_length,
                const int q_num_head,
                const int kv_num_head,
                const int dim_head,
                const int max_seq_len,
                const int rotary_emb_dims,
                const bool neox_rotary_style,
                const phi::DenseTensor* cache_k_quant_scales,
                const phi::DenseTensor* cache_v_quant_scales,
                const phi::DenseTensor* cache_k_dequant_scales,
                const phi::DenseTensor* cache_v_dequant_scales,
                const phi::DenseTensor* shift,
                const phi::DenseTensor* smooth) {
  using params_type = Block_attn_params<T, out_type, cachenbits>;
  params_type params;
  memset(&params, 0, sizeof(params));

  params.qkv_input = const_cast<T*>(qkv_tensor.data<T>());
  if (qkv_bias_tensor) {
    params.qkv_bias = const_cast<T*>(qkv_bias.data<T>());
  } else {
    params.qkv_bias = nullptr;
  }
  params.attn_out = out_tensor->data<typename params_type::out_type>();

  params.seq_lens = const_cast<int*>(sequence_lengths_tensor->data<int>());
  if (cum_offsets_tensor) {
    params.cum_offsets = const_cast<int*>(cum_offsets_tensor->data<int>());
  } else {
    params.cum_offsets = nullptr;
  }
  params.block_table = const_cast<int*>(block_tables->data<int>());
  if (rotary_tensor) {
    params.rotary_emb = const_cast<float*>(rotary_tensor->data<float>());
  } else {
    params.rotary_emb = nullptr;
  }
  params.inv_compression_ratio = 1.0f;  // Default as 1.0
  params.rope_theta = 10000.0;
  params.rotary_emb_dims = rotary_emb_dims;
  params.max_input_length = max_input_length;
  params.head_num = q_num_head;
  params.kv_head_num = kv_num_head;
  params.max_num_blocks_per_seq = max_num_blocks_per_seq;

  params.batch_size = batch_size;
  assert(dim_head == 128);
  params.inv_sqrt_dh = 1.0f / std::sqrt(dim_head);
  // 多减1是因为计算新的qk
  const uint32_t max_num_partitions =
      (max_seq_len + max_seq_per_block) / max_seq_per_block;
  params.max_num_partitions = max_num_partitions;
  phi::DenseTensor maxs, sums, partition_attn_out;
  maxs.Resize({{batch_size, q_num_head, (max_num_partitions + 3) / 4 * 4}});
  sums.Resize({{batch_size, q_num_head, (max_num_partitions + 3) / 4 * 4}});
  partition_attn_out.Resize(
      {{batch_size, max_num_partitions, q_num_head, dim_head}});
  params.partition_attn_out = dev_ctx.template Alloc<T>(&partition_attn_out);

  params.maxs = dev_ctx.template Alloc<float>(&maxs);
  params.sums = dev_ctx.template Alloc<float>(&sums);

  using cache_traits = CacheKV_quant_traits<T, cachenbits>;
  if constexpr (cachenbits == 16) {
    params.cache_k = const_cast<T*>(k_cache->data<T>());
    params.cache_v = const_cast<T*>(v_cache->data<T>());
    params.cache_k_dequant_scale = nullptr;
    params.cache_v_dequant_scale = nullptr;
    params.cache_k_quant_scale = nullptr;
    params.cache_v_quant_scale = nullptr;
    WriteC16DecoderCacheKvHdim128<T> writeDecoderCacheKV;
    writeDecoderCacheKV(params.qkv_input,
                        params.qkv_bias,
                        params.rotary_emb,
                        params.block_table,
                        params.seq_lens,
                        params.cache_k,
                        params.cache_v,
                        bsz,
                        head_num,
                        kv_head_num,
                        head_dim,
                        params.max_num_blocks_per_seq,
                        dev_ctx.stream());
  } else {
    params.cache_k = const_cast<uint8_t*>(k_cache->data<uint8_t>());
    params.cache_v = const_cast<uint8_t*>(v_cahce->data<uint8_t>());
    params.cache_k_dequant_scale = cache_k_dequant_scales->data<float>();
    params.cache_v_dequant_scale = cache_v_dequant_scales->data<float>();
    params.cache_k_quant_scale = cache_k_quant_scales->data<float>();
    params.cache_v_quant_scale = cache_v_quant_scales->data<float>();
    WriteDecoderCacheKvHdim128<T, cachenbits> writeDecoderCacheKV;
    writeDecoderCacheKV(params.qkv_input,
                        params.qkv_bias,
                        params.rotary_emb,
                        params.block_table,
                        params.seq_lens,
                        params.cache_k,
                        params.cache_v,
                        params.cache_k_quant_scale,
                        params.cache_v_quant_scale,
                        bsz,
                        head_num,
                        kv_head_num,
                        head_dim,
                        params.max_num_blocks_per_seq,
                        dev_ctx.stream());
  }
  run_block_attn_hdim128<T, cache_traits, params_type, max_seq_per_block>(
      params, dev_ctx.stream());
}

template <typename T>
void blha_tc(const phi::GPUContext& dev_ctx,
             const phi::DenseTensor& qkv_tensor,
             const phi::DenseTensor* qkv_bias_tensor,
             const phi::DenseTensor* block_tables,
             const phi::DenseTensor* cum_offsets_tensor,
             const phi::DenseTensor* sequence_lengths_tensor,
             const phi::DenseTensor* rotary_tensor,
             phi::DenseTensor* k_cache,
             phi::DenseTensor* v_cache,
             phi::DenseTensor* out_tensor,
             const int batch_size,
             const int max_num_blocks_per_seq,
             const int block_size,
             const int max_input_length,
             const int q_num_head,
             const int kv_num_head,
             const int dim_head,
             const int max_seq_len,
             const int rotary_emb_dims,
             const bool neox_rotary_style = false,
             const phi::DenseTensor* cache_k_quant_scales = nullptr,
             const phi::DenseTensor* cache_v_quant_scales = nullptr,
             const phi::DenseTensor* cache_k_dequant_scales = nullptr,
             const phi::DenseTensor* cache_v_dequant_scales = nullptr,
             const phi::DenseTensor* shift = nullptr,
             const phi::DenseTensor* smooth = nullptr,
             int use_cachekv_int8 = 0) {
  constexpr int max_seq_per_block = 128;
  assert(use_cachekv_int8 == 0 || use_cachekv_int8 == 1);
  auto set_params_func = set_params<T, Context, int8_t, max_seq_per_block, 16>;

  dev_ctx.template Alloc<T>(attn_out);
  if (use_cachekv_int8 > 0) {
    set_params_func = set_params<T, Context, T, max_seq_per_block, 8>;
  } else {
    set_params_func = set_params<T, Context, T, max_seq_per_block, 16>;
  }

  set_params_func(dev_ctx,
                  qkv_tensor,
                  qkv_bias_tensor,
                  block_tables,
                  cum_offsets_tensor,
                  sequence_lengths_tensor,
                  rotary_tensor,
                  k_cache,
                  v_cache,
                  out_tensor,
                  batch_size,
                  max_num_blocks_per_seq,
                  max_input_length,
                  q_num_head,
                  kv_num_head,
                  dim_head,
                  max_seq_len,
                  rotary_emb_dims,
                  neox_rotary_style,
                  cache_k_quant_scales,
                  cache_v_quant_scales,
                  cache_k_dequant_scales,
                  cache_v_dequant_scales,
                  shift,
                  smooth);
}

}  // namespace fusion
}  // namespace phi
