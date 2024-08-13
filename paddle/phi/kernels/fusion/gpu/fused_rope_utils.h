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
constexpr int kDefaultRotaryBase = 10000;
constexpr float Epsilon = 1e-7;
template <typename T, typename MPType, int VecSize>
using VectorizedFusedRopeCudaKernelFunc =
    void (*)(phi::Array<const T*, 3> ins_data,
             phi::Array<const T*, 2> sin_cos_data,
             const int64_t* position_ids_data,
             bool flag_sin_cos,
             int sign,
             int64_t batch_size,
             int64_t seq_len,
             int64_t num_heads,
             int64_t head_dim,
             int64_t batch_stride,
             int64_t seq_stride,
             int num_inputs,
             MPType div_c,
             float rotary_emb_base,
             phi::Array<T*, 3> outs_data);

template <typename T, typename MPType, int VecSize = 2>
__device__ __forceinline__ void get_sin_cos_by_passed_values(
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    int64_t index,
    int64_t batch_size,
    int64_t seq_len,
    int64_t head_dim,
    int64_t batch_stride,
    int64_t seq_stride,
    MPType* out_sin,
    MPType* out_cos) {
#pragma unroll
  for (int64_t nx = 0; nx < VecSize; ++nx) {
    int64_t pos_seq_ori = (index + nx) / seq_stride % seq_len;
    int64_t pos_seq;
    if (position_ids_data) {
      int64_t pos_bs = (index + nx) / batch_stride % batch_size;
      int64_t index_ids = pos_bs * seq_len + pos_seq_ori;
      pos_seq = position_ids_data[index_ids];
    } else {
      pos_seq = pos_seq_ori;
    }
    int64_t pos_head = (index + nx) % head_dim;
    int64_t index_sc = pos_seq * head_dim + pos_head;
    const T* sin_input = sin_cos_data[0] + index_sc;
    const T* cos_input = sin_cos_data[1] + index_sc;

    out_sin[nx] = static_cast<MPType>(sin_input[0]);
    out_cos[nx] = static_cast<MPType>(cos_input[0]);
  }
}

template <typename MPType, int VecSize = 2>
__device__ __forceinline__ void get_sin_cos_by_rotary_base(
    int64_t index,
    int64_t seq_len,
    int64_t head_dim,
    int64_t seq_stride,
    float rotary_emb_base,
    MPType div_c,
    MPType* out_sin,
    MPType* out_cos) {
#pragma unroll
  for (int nx = 0; nx < VecSize; ++nx) {
    // get sin_index and cos_index
    int64_t pos_seq = (index + nx) / seq_stride % seq_len;

    MPType idx = static_cast<MPType>(((index + nx) % head_dim) / 2 * 2.0);
    MPType indicses = static_cast<MPType>(1) /
                      pow(static_cast<MPType>(rotary_emb_base), idx * div_c);
    MPType value = pos_seq * indicses;
    out_sin[nx] = sin(value);
    out_cos[nx] = cos(value);
  }
}

template <typename T, typename MPType, int VecSize = 2, int ROTARY_BASE = -1>
struct VectorizedGetSinCos {
  static __device__ void run(phi::Array<const T*, 2> sin_cos_data,
                             const int64_t* position_ids_data,
                             bool flag_sin_cos,
                             int64_t index,
                             int64_t batch_size,
                             int64_t seq_len,
                             int64_t num_heads,
                             int64_t head_dim,
                             int64_t batch_stride,
                             int64_t seq_stride,
                             MPType div_c,
                             float rotary_emb_base,
                             MPType* out_sin,
                             MPType* out_cos) {
    if (flag_sin_cos) {
      get_sin_cos_by_passed_values(sin_cos_data,
                                   position_ids_data,
                                   index,
                                   batch_size,
                                   seq_len,
                                   head_dim,
                                   batch_stride,
                                   seq_stride,
                                   out_sin,
                                   out_cos);
    } else {
      get_sin_cos_by_rotary_base(index,
                                 seq_len,
                                 head_dim,
                                 seq_stride,
                                 rotary_emb_base,
                                 div_c,
                                 out_sin,
                                 out_cos);
    }
  }
};

// NOTE(zhengzhonghui): make rotary_emb_base as a parameter will cause the cuda
// kernel slower down than before, so partial specialization for
// kDefaultRotaryBase=10000 to avoid the slowdown.
template <typename T, typename MPType, int VecSize>
struct VectorizedGetSinCos<T, MPType, VecSize, kDefaultRotaryBase> {
  static __device__ void run(phi::Array<const T*, 2> sin_cos_data,
                             const int64_t* position_ids_data,
                             bool flag_sin_cos,
                             int64_t index,
                             int64_t batch_size,
                             int64_t seq_len,
                             int64_t num_heads,
                             int64_t head_dim,
                             int64_t batch_stride,
                             int64_t seq_stride,
                             MPType div_c,
                             float rotary_emb_base,
                             MPType* out_sin,
                             MPType* out_cos) {
    MPType* sin_value = out_sin;
    MPType* cos_value = out_cos;

    if (flag_sin_cos) {
      get_sin_cos_by_passed_values(sin_cos_data,
                                   position_ids_data,
                                   index,
                                   batch_size,
                                   seq_len,
                                   head_dim,
                                   batch_stride,
                                   seq_stride,
                                   out_sin,
                                   out_cos);
    } else {
#pragma unroll
      for (int nx = 0; nx < VecSize; ++nx) {
        // get sin_index and cos_index
        int64_t pos_seq = (index + nx) / seq_stride % seq_len;

        MPType idx = static_cast<MPType>(((index + nx) % head_dim) / 2 * 2.0);
        MPType indicses =
            static_cast<MPType>(1) /
            pow(static_cast<MPType>(kDefaultRotaryBase), idx * div_c);

        MPType value = pos_seq * indicses;
        sin_value[nx] = sin(value);
        cos_value[nx] = cos(value);
      }
    }
  }
};

template <typename T, typename MPType, int VecSize = 2>
__device__ __forceinline__ void rotate_every_two(
    phi::Array<const T*, 3> ins_data,
    int num_inputs,
    int64_t index,
    int sign,
    MPType* sin_value,
    MPType* cos_value,
    phi::Array<T*, 3> outs_data) {
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;
  MPType result[VecSize];
  T store[VecSize];
#pragma unroll
  for (int iter = 0; iter < 3; iter++) {
    if (iter >= num_inputs) break;
    const T* input = ins_data[iter] + index;
    VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
    for (int nx = 0; nx < kVectorsPerThread; ++nx) {
      int pr_index = nx * 2;
      int ls_index = pr_index + 1;

      MPType p0 = static_cast<MPType>(input[pr_index]);
      MPType p1 = static_cast<MPType>(input[ls_index]);

      if (sign == 1) {
        result[pr_index] = cos_value[pr_index] * p0;
        result[pr_index] -= sin_value[pr_index] * p1;

        result[ls_index] = sin_value[ls_index] * p0;
        result[ls_index] += cos_value[ls_index] * p1;
      } else if (sign == -1) {
        result[pr_index] = cos_value[pr_index] * p0 + sin_value[ls_index] * p1;
        result[ls_index] = cos_value[ls_index] * p1 - sin_value[pr_index] * p0;
      }

      store[pr_index] = static_cast<T>(result[pr_index]);
      store[ls_index] = static_cast<T>(result[ls_index]);
    }
    out[0] = *(reinterpret_cast<VecType*>(store));
  }
}

template <typename T, typename MPType, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateEveryTwoKernel(
    phi::Array<const T*, 3> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t batch_stride,
    int64_t seq_stride,
    int num_inputs,
    MPType div_c,
    float rotary_emb_base,
    phi::Array<T*, 3> outs_data) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];

  if (fabs(rotary_emb_base - static_cast<float>(kDefaultRotaryBase)) <
      Epsilon) {
    for (; index < size; index += stride) {
      VectorizedGetSinCos<T, MPType, VecSize, kDefaultRotaryBase>::run(
          sin_cos_data,
          position_ids_data,
          flag_sin_cos,
          index,
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          batch_stride,
          seq_stride,
          div_c,
          rotary_emb_base,
          sin_value,
          cos_value);
      rotate_every_two<T, MPType, VecSize>(
          ins_data, num_inputs, index, sign, sin_value, cos_value, outs_data);
    }
  } else {
    for (; index < size; index += stride) {
      VectorizedGetSinCos<T, MPType, VecSize>::run(sin_cos_data,
                                                   position_ids_data,
                                                   flag_sin_cos,
                                                   index,
                                                   batch_size,
                                                   seq_len,
                                                   num_heads,
                                                   head_dim,
                                                   batch_stride,
                                                   seq_stride,
                                                   div_c,
                                                   rotary_emb_base,
                                                   sin_value,
                                                   cos_value);
      rotate_every_two<T, MPType, VecSize>(
          ins_data, num_inputs, index, sign, sin_value, cos_value, outs_data);
    }
  }
}

template <typename T, typename MPType, int VecSize = 2>
__device__ __forceinline__ void rotate_half(phi::Array<const T*, 3> ins_data,
                                            int num_inputs,
                                            int64_t head_dim,
                                            int64_t index,
                                            int sign,
                                            MPType* sin_value,
                                            MPType* cos_value,
                                            phi::Array<T*, 3> outs_data) {
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;
  int64_t stride_r = head_dim / 2;
#pragma unroll
  for (int iter = 0; iter < 3; iter++) {
    if (iter >= num_inputs) break;
    // get value_index and rotate_half_index
    int64_t index_v = index;
    int64_t index_r =
        (index % head_dim) < stride_r ? (index + stride_r) : (index - stride_r);
    MPType sign_r = (index % head_dim) < stride_r ? static_cast<MPType>(-1)
                                                  : static_cast<MPType>(1);
    const T* input_v = ins_data[iter] + index_v;
    const T* input_r = ins_data[iter] + index_r;
    VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
    for (int nx = 0; nx < VecSize; ++nx) {
      MPType p0 = static_cast<MPType>(input_v[nx]);
      MPType p1 = static_cast<MPType>(input_r[nx]);

      result[nx] = cos_value[nx] * p0 + sign * sign_r * sin_value[nx] * p1;

      store[nx] = static_cast<T>(result[nx]);
    }
    out[0] = *(reinterpret_cast<VecType*>(store));
  }
}

template <typename T, typename MPType, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateHalfKernel(
    phi::Array<const T*, 3> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t batch_stride,
    int64_t seq_stride,
    int num_inputs,
    MPType div_c,
    float rotary_emb_base,
    phi::Array<T*, 3> outs_data) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];

  if (fabs(rotary_emb_base - static_cast<float>(kDefaultRotaryBase)) <
      Epsilon) {
    for (; index < size; index += stride) {
      VectorizedGetSinCos<T, MPType, VecSize, kDefaultRotaryBase>::run(
          sin_cos_data,
          position_ids_data,
          flag_sin_cos,
          index,
          batch_size,
          seq_len,
          num_heads,
          head_dim,
          batch_stride,
          seq_stride,
          div_c,
          rotary_emb_base,
          sin_value,
          cos_value);
      rotate_half<T, MPType, VecSize>(ins_data,
                                      num_inputs,
                                      head_dim,
                                      index,
                                      sign,
                                      sin_value,
                                      cos_value,
                                      outs_data);
    }
  } else {
    VectorizedGetSinCos<T, MPType, VecSize>::run(sin_cos_data,
                                                 position_ids_data,
                                                 flag_sin_cos,
                                                 index,
                                                 batch_size,
                                                 seq_len,
                                                 num_heads,
                                                 head_dim,
                                                 batch_stride,
                                                 seq_stride,
                                                 div_c,
                                                 rotary_emb_base,
                                                 sin_value,
                                                 cos_value);
    rotate_half<T, MPType, VecSize>(ins_data,
                                    num_inputs,
                                    head_dim,
                                    index,
                                    sign,
                                    sin_value,
                                    cos_value,
                                    outs_data);
  }
}

}  // namespace fusion
}  // namespace phi
