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

#include "paddle/phi/kernels/write_cache_kv_kernel.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/kernel_registry.h"

namespace {

constexpr int VEC_16B = 16;

template <typename T>
inline __device__ __host__ T div_up(T m, T n) {
  return (m + n - 1) / n;
}

template <typename T>
__global__ void write_cache_k_kernel(T *cache_k,
                                     const T *k,
                                     const int *seq_lens,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int len = seq_lens ? seq_lens[bi] : seq_len;
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto k_src = reinterpret_cast<const uint4 *>(
      k + bi * num_head * seq_len * dim_head + hi * seq_len * dim_head);
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  auto k_dst = reinterpret_cast<uint4 *>(
      cache_k + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // vec size
  int dim_head_div_x = dim_head / X_ELEMS;

  // FIXME(wangxi): num_head is not need?
  // if (out_idx >= num_head * dim_head_div_x * max_seq_len) return;
  if (out_idx >= dim_head_div_x * max_seq_len) return;

  int idx = out_idx;
  const int k_seq_len_id = idx % max_seq_len;
  // idx = (idx - k_seq_len_id) / max_seq_len;
  idx = idx / max_seq_len;
  const int k_vec_id = idx % dim_head_div_x;

  if (k_seq_len_id < len) {
    k_dst[out_idx] = k_src[k_seq_len_id * dim_head_div_x + k_vec_id];
  }
}

template <typename T>
__global__ void write_cache_v_kernel(T *cache_v,
                                     const T *v,
                                     const int *seq_lens,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int len = seq_lens ? seq_lens[bi] : seq_len;
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto v_src = reinterpret_cast<const uint4 *>(
      v + bi * num_head * seq_len * dim_head + hi * seq_len * dim_head);
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  auto v_dst = reinterpret_cast<uint4 *>(
      cache_v + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);
  const int dim_head_div_x = dim_head / X_ELEMS;

  if (idx >= dim_head_div_x * len) return;

  v_dst[idx] = v_src[idx];
}

}  // namespace

namespace phi {

template <typename T, typename Context>
void WriteCacheKVKernel(const Context &ctx,
                        const DenseTensor &input_k,
                        const DenseTensor &input_v,
                        const DenseTensor &cache_kv,
                        const DenseTensor &sequence_lengths,
                        DenseTensor *cache_kv_out) {
  ctx.template Alloc<T>(cache_kv_out);

  using DataType_ = typename PDDataTypeTraits<T>::DataType;

  const int64_t bsz = input_k.dims()[0];
  const int64_t seq_len = input_k.dims()[2];
  const int64_t cache_bsz = cache_kv.dims()[1];
  const int64_t num_head = cache_kv.dims()[2];
  const int64_t dim_head = cache_kv.dims()[4];

  const DataType_ *k_ptr =
      reinterpret_cast<const DataType_ *>(input_k.data<T>());
  const DataType_ *v_ptr =
      reinterpret_cast<const DataType_ *>(input_v.data<T>());

  // [2, bsz, num_head, max_seq_len, head_dim]
  const int64_t max_seq_len = cache_kv.dims()[3];
  DataType_ *cache_kv_data =
      reinterpret_cast<DataType_ *>(const_cast<T *>(cache_kv.data<T>()));

  int64_t cache_k_size = cache_bsz * num_head * max_seq_len * dim_head;

  DataType_ *cache_k_ptr = cache_kv_data;
  DataType_ *cache_v_ptr = cache_kv_data + cache_k_size;

  constexpr int block_sz = 128;
  constexpr int x = VEC_16B / sizeof(DataType_);

  assert(dim_head % x == 0);

  int max_size = max_seq_len * dim_head / x;
  int size = seq_len * dim_head / x;
  dim3 grid(div_up(max_size, block_sz), bsz, num_head);
  dim3 grid_v(div_up(size, block_sz), bsz, num_head);

  // transpose [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  write_cache_k_kernel<<<grid, block_sz, 0, ctx.stream()>>>(
      cache_k_ptr,
      k_ptr,
      sequence_lengths.data<int>(),
      num_head,
      dim_head,
      seq_len,
      max_seq_len);

  // copy [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  write_cache_v_kernel<<<grid_v, block_sz, 0, ctx.stream()>>>(
      cache_v_ptr,
      v_ptr,
      sequence_lengths.data<int>(),
      num_head,
      dim_head,
      seq_len,
      max_seq_len);
}

}  // namespace phi

PD_REGISTER_KERNEL(write_cache_kv,
                   GPU,
                   ALL_LAYOUT,
                   phi::WriteCacheKVKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
