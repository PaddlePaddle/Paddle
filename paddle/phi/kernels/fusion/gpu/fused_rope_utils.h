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
__device__ void set_cin_cos_shared_mem(const T* sin,
                                       const T* cos,
                                       const int64_t* position_ids,
                                       const bool flag_sin_cos,
                                       const float rotary_emb_base,
                                       const IndexT seq_len,
                                       const IndexT s_id,
                                       const IndexT b_id,
                                       const IndexT d,
                                       float* shared_mem_sin,
                                       float* shared_mem_cos) {
  IndexT tid = static_cast<IndexT>(threadIdx.x) * blockDim.y + threadIdx.y;
  for (IndexT i = tid; i < d; i += blockDim.x * blockDim.y) {
    int64_t pos = s_id;
    if (position_ids) {
      pos = position_ids[b_id * seq_len + s_id];
    }

    if (flag_sin_cos) {
      shared_mem_sin[i] = static_cast<float>(sin[pos * d + i]);
      shared_mem_cos[i] = static_cast<float>(cos[pos * d + i]);
    } else {
      float idx = static_cast<float>((i / 2) * 2);
      float inv_freq =
          1.0f / powf(rotary_emb_base, idx / static_cast<float>(d));
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

  extern __shared__ float shared_mem_cos_sin[];
  float* shared_mem_cos = shared_mem_cos_sin;
  float* shared_mem_sin = shared_mem_cos_sin + d;

  set_cin_cos_shared_mem<T>(sin,
                            cos,
                            position_ids,
                            flag_sin_cos,
                            rotary_emb_base,
                            seq_len,
                            s_id,
                            b_id,
                            d,
                            shared_mem_sin,
                            shared_mem_cos);

#pragma unroll
  for (IndexT h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (IndexT d_id = threadIdx.x; d_id < d; d_id += blockDim.x) {
      float v_cos = shared_mem_cos[d_id];
      float v_sin = shared_mem_sin[d_id];
      IndexT offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      IndexT offset_dst =
          offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = static_cast<float>(src[offset_src]);
      float v_src_rotate;
      if (!use_neox_rotary_style) {
        v_src_rotate =
            (d_id + d / 2 < d)
                ? -static_cast<float>(src[offset_src + (d / 2) * stride_d])
                : static_cast<float>(src[offset_src + (d / 2 - d) * stride_d]);
      } else {
        v_src_rotate = (d_id % 2 == 0)
                           ? -static_cast<float>(src[offset_src + stride_d])
                           : static_cast<float>(src[offset_src - stride_d]);
      }
      dst[offset_dst] = static_cast<T>(v_src * v_cos + v_src_rotate * v_sin);
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

  extern __shared__ float shared_mem_cos_sin[];
  float* shared_mem_cos = shared_mem_cos_sin;
  float* shared_mem_sin = shared_mem_cos_sin + d;

  set_cin_cos_shared_mem<T>(sin,
                            cos,
                            position_ids,
                            flag_sin_cos,
                            rotary_emb_base,
                            seq_len,
                            s_id,
                            b_id,
                            d,
                            shared_mem_sin,
                            shared_mem_cos);

#pragma unroll
  for (IndexT h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (IndexT d_id = threadIdx.x; d_id < d; d_id += blockDim.x) {
      IndexT offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      IndexT offset_dst =
          offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = static_cast<float>(src[offset_src]);
      float v_cos = shared_mem_cos[d_id];
      float v_src_rotate, v_sin;
      if (!use_neox_rotary_style) {
        if (d_id + d / 2 < d) {
          v_src_rotate =
              static_cast<float>(src[offset_src + (d / 2) * stride_d]);
          v_sin = shared_mem_sin[d_id + d / 2];
        } else {
          v_src_rotate =
              static_cast<float>(src[offset_src + (d / 2 - d) * stride_d]);
          v_sin = -shared_mem_sin[d_id + d / 2 - d];
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
}

}  // namespace fusion
}  // namespace phi
