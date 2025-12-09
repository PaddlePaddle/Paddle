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

#pragma once

#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace phi {
namespace fusion {

template <typename T, typename IndexT>
__device__ void set_sin_cos_shared_mem(const T* sin,
                                       const T* cos,
                                       const int64_t* position_ids,
                                       const bool flag_sin_cos,
                                       const float rotary_emb_base,
                                       const IndexT seq_len,
                                       const IndexT s_id,
                                       const IndexT b_id,
                                       const IndexT d,
                                       const IndexT d2,
                                       float* shared_mem_sin,
                                       float* shared_mem_cos) {
  IndexT tid = static_cast<IndexT>(threadIdx.x) * blockDim.y + threadIdx.y;
  for (IndexT i = tid; i < d2; i += blockDim.x * blockDim.y) {
    int64_t pos = s_id;
    if (position_ids) {
      pos = position_ids[b_id * seq_len + s_id];
    }

    if (flag_sin_cos) {
      shared_mem_sin[i] = static_cast<float>(sin[pos * d2 + i]);
      shared_mem_cos[i] = static_cast<float>(cos[pos * d2 + i]);
    } else {
      float idx = static_cast<float>((i / 2) * 2);
      float inv_freq =
          1.0f / powf(rotary_emb_base, idx / static_cast<float>(d2));
      float freq = static_cast<float>(pos) * inv_freq;
      sincosf(freq, &shared_mem_sin[i], &shared_mem_cos[i]);
    }
  }
  __syncthreads();
}

template <typename T, typename IndexT>
__global__ void FusedRopeKernelImpl(const T* src,
                                    const T* sin,
                                    const T* cos,
                                    T* dst,
                                    const int64_t* position_ids,
                                    const bool flag_sin_cos,
                                    const bool use_neox_rotary_style,
                                    const IndexT h,
                                    const IndexT d,
                                    const IndexT d2,
                                    const IndexT stride_s,
                                    const IndexT stride_b,
                                    const IndexT stride_h,
                                    const IndexT stride_d,
                                    const IndexT o_stride_s,
                                    const IndexT o_stride_b,
                                    const IndexT o_stride_h,
                                    const IndexT o_stride_d,
                                    const float rotary_emb_base,
                                    const IndexT seq_len) {
  IndexT s_id = blockIdx.x;
  IndexT b_id = blockIdx.y;

  IndexT offset_block = s_id * stride_s + b_id * stride_b;
  IndexT offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;

  extern __shared__ float shared_mem[];
  float* shared_mem_cos = shared_mem;
  float* shared_mem_sin = shared_mem + d2;

  set_sin_cos_shared_mem<T>(sin,
                            cos,
                            position_ids,
                            flag_sin_cos,
                            rotary_emb_base,
                            seq_len,
                            s_id,
                            b_id,
                            d,
                            d2,
                            shared_mem_sin,
                            shared_mem_cos);

#pragma unroll
  for (IndexT h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (IndexT d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      float v_cos = shared_mem_cos[d_id];
      float v_sin = shared_mem_sin[d_id];
      IndexT offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      IndexT offset_dst =
          offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = static_cast<float>(src[offset_src]);
      float v_src_rotate;
      if (!use_neox_rotary_style) {
        v_src_rotate =
            (d_id + d2 / 2 < d2)
                ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
                : static_cast<float>(
                      src[offset_src + (d2 / 2 - d2) * stride_d]);
      } else {
        v_src_rotate = (d_id % 2 == 0)
                           ? -static_cast<float>(src[offset_src + stride_d])
                           : static_cast<float>(src[offset_src - stride_d]);
      }
      dst[offset_dst] = static_cast<T>(v_src * v_cos + v_src_rotate * v_sin);
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
#pragma unroll
      for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
        int offset_dst =
            offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
        dst[offset_dst] = src[offset_src];
      }
    }
  }
}

template <typename T, typename IndexT>
__global__ void FusedRopeGradKernelImpl(const T* src,
                                        const T* sin,
                                        const T* cos,
                                        T* dst,
                                        const int64_t* position_ids,
                                        const bool flag_sin_cos,
                                        const bool use_neox_rotary_style,
                                        const IndexT h,
                                        const IndexT d,
                                        const IndexT d2,
                                        const IndexT stride_s,
                                        const IndexT stride_b,
                                        const IndexT stride_h,
                                        const IndexT stride_d,
                                        const IndexT o_stride_s,
                                        const IndexT o_stride_b,
                                        const IndexT o_stride_h,
                                        const IndexT o_stride_d,
                                        const float rotary_emb_base,
                                        const IndexT seq_len) {
  IndexT s_id = blockIdx.x;
  IndexT b_id = blockIdx.y;

  IndexT offset_block = s_id * stride_s + b_id * stride_b;
  IndexT offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;

  extern __shared__ float shared_mem[];
  float* shared_mem_cos = shared_mem;
  float* shared_mem_sin = shared_mem + d2;

  set_sin_cos_shared_mem<T>(sin,
                            cos,
                            position_ids,
                            flag_sin_cos,
                            rotary_emb_base,
                            seq_len,
                            s_id,
                            b_id,
                            d,
                            d2,
                            shared_mem_sin,
                            shared_mem_cos);

#pragma unroll
  for (IndexT h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (IndexT d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      IndexT offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      IndexT offset_dst =
          offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = static_cast<float>(src[offset_src]);
      float v_cos = shared_mem_cos[d_id];
      float v_src_rotate, v_sin;
      if (!use_neox_rotary_style) {
        if (d_id + d2 / 2 < d2) {
          v_src_rotate =
              static_cast<float>(src[offset_src + (d2 / 2) * stride_d]);
          v_sin = shared_mem_sin[d_id + d2 / 2];
        } else {
          v_src_rotate =
              static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
          v_sin = -shared_mem_sin[d_id + d2 / 2 - d2];
        }
      } else {
        if (d_id % 2 == 0) {
          v_src_rotate = static_cast<float>(src[offset_src + stride_d]);
          v_sin = shared_mem_sin[d_id + 1];
        } else {
          v_src_rotate = static_cast<float>(src[offset_src - stride_d]);
          v_sin = -shared_mem_sin[d_id - 1];
        }
      }
      dst[offset_dst] = static_cast<T>(v_src * v_cos + v_src_rotate * v_sin);
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
#pragma unroll
      for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
        int offset_dst =
            offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
        dst[offset_dst] = src[offset_src];
      }
    }
  }
}

template <typename T, typename IndexT>
using FusedRopeKernelFunc = void (*)(const T*,
                                     const T*,
                                     const T*,
                                     T*,
                                     const int64_t*,
                                     const bool,
                                     const bool,
                                     const IndexT,
                                     const IndexT,
                                     const IndexT,
                                     const IndexT,
                                     const IndexT,
                                     const IndexT,
                                     const IndexT,
                                     const IndexT,
                                     const IndexT,
                                     const IndexT,
                                     const IndexT,
                                     const float,
                                     const IndexT);

template <typename T>
void FusedRopeKernelLauncher(const T* src,
                             const T* sin,
                             const T* cos,
                             T* dst,
                             FusedRopeKernelFunc<T, int> kernel_int32,
                             FusedRopeKernelFunc<T, int64_t> kernel_int64,
                             const int64_t* position_ids,
                             const bool flag_sin_cos,
                             const bool use_neox_rotary_style,
                             const int64_t h,
                             const int64_t d,
                             const int64_t d2,
                             const int64_t stride_s,
                             const int64_t stride_b,
                             const int64_t stride_h,
                             const int64_t stride_d,
                             const int64_t o_stride_s,
                             const int64_t o_stride_b,
                             const int64_t o_stride_h,
                             const int64_t o_stride_d,
                             const float rotary_emb_base,
                             const int64_t seq_len,
                             const int64_t batch_size,
                             const int64_t numel,
                             gpuStream_t stream) {
  const int64_t warps_per_block = h < 16 ? 4 : 8;
  dim3 grid(seq_len, batch_size);
  dim3 block(32, warps_per_block);  // 32 threads per warp
  size_t shared_mem_size = 2 * d2 * sizeof(float);

  if (numel <= std::numeric_limits<int>::max()) {
    kernel_int32<<<grid, block, shared_mem_size, stream>>>(
        src,
        sin,
        cos,
        dst,
        position_ids,
        flag_sin_cos,
        use_neox_rotary_style,
        h,
        d,
        d2,
        stride_s,
        stride_b,
        stride_h,
        stride_d,
        o_stride_s,
        o_stride_b,
        o_stride_h,
        o_stride_d,
        rotary_emb_base,
        seq_len);
  } else {
    kernel_int64<<<grid, block, shared_mem_size, stream>>>(
        src,
        sin,
        cos,
        dst,
        position_ids,
        flag_sin_cos,
        use_neox_rotary_style,
        h,
        d,
        d2,
        stride_s,
        stride_b,
        stride_h,
        stride_d,
        o_stride_s,
        o_stride_b,
        o_stride_h,
        o_stride_d,
        rotary_emb_base,
        seq_len);
  }
}

}  // namespace fusion
}  // namespace phi
