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

// Return the number of heads for key and value. If both key and value are None,
// return num_heads of query.
int64_t GetNumHeadsOfKV(const paddle::optional<DenseTensor>& k,
                        const paddle::optional<DenseTensor>& v,
                        const phi::Array<int64_t, 3>& inputs_num_heads,
                        int num_inputs) {
  bool have_same_num_heads = true;
  auto num_heads_kv = inputs_num_heads[0];
  for (int i = 1; i < num_inputs; ++i) {
    if (num_heads_kv != inputs_num_heads[i]) {
      have_same_num_heads = false;
      num_heads_kv = inputs_num_heads[i];
      break;
    }
  }

  if (!have_same_num_heads) {
    // Multi Query Attention (MQA) or Group Query Attention (GQA)
    // check query, key, value shape
    PADDLE_ENFORCE_EQ(
        (inputs_num_heads[0] != inputs_num_heads[num_inputs - 1]) &&
            (inputs_num_heads[0] % inputs_num_heads[num_inputs - 1] == 0),
        true,
        phi::errors::InvalidArgument(
            "The MQA or GQA mode is entered, when the number of heads of qkv "
            "is not exactly the same two by two. This mode requires "
            "num_heads of q to be divisible by k,v."
            "But recieved num_heads of q is %d, num_heads of k,v is %d",
            inputs_num_heads[0],
            inputs_num_heads[num_inputs - 1]));

    if (k.get_ptr() && v.get_ptr()) {
      PADDLE_ENFORCE_EQ(
          inputs_num_heads[1] == inputs_num_heads[2],
          true,
          phi::errors::InvalidArgument(
              "The num_heads of k must be equal to the num_heads of v when v "
              "is not none."
              "But recieved num_heads of k is %d, num_heads of v is %d",
              inputs_num_heads[1],
              inputs_num_heads[2]));
    }
  }

  return num_heads_kv;
}

template <typename T, typename MPType, int NInputs, int VecSize>
using VectorizedFusedRopeCudaKernelFunc =
    void (*)(phi::Array<const T*, NInputs> ins_data,
             phi::Array<const T*, 2> sin_cos_data,
             const int64_t* position_ids_data,
             bool flag_sin_cos,
             int sign,
             int64_t batch_size,
             int64_t seq_len,
             int64_t num_heads,
             int64_t head_dim,
             phi::Array<T*, NInputs> outs_data,
             int num_inputs,
             MPType div_c);

template <typename T, typename MPType, int VecSize = 2>
__device__ void VectorizedGetSinCos(phi::Array<const T*, 2> sin_cos_data,
                                    const int64_t* position_ids_data,
                                    bool flag_sin_cos,
                                    int64_t index,
                                    int64_t seq_len,
                                    int64_t num_heads,
                                    int64_t head_dim,
                                    MPType* out_sin,
                                    MPType* out_cos,
                                    MPType div_c) {
  MPType* sin_value = out_sin;
  MPType* cos_value = out_cos;

  if (flag_sin_cos) {
#pragma unroll
    for (int64_t nx = 0; nx < VecSize; ++nx) {
      int64_t index_wc = (index + nx) % (seq_len * num_heads * head_dim);
      int64_t pos_seq_ori = index_wc / (num_heads * head_dim);
      int64_t pos_seq;
      if (position_ids_data) {
        int64_t pos_bs = (index + nx) / (seq_len * num_heads * head_dim);
        int64_t index_ids = pos_bs * seq_len + pos_seq_ori;
        pos_seq = position_ids_data[index_ids];
      } else {
        pos_seq = pos_seq_ori;
      }
      int64_t pos_head = index_wc % head_dim;
      int64_t index_sc = pos_seq * head_dim + pos_head;
      const T* sin_input = sin_cos_data[0] + index_sc;
      const T* cos_input = sin_cos_data[1] + index_sc;

      sin_value[nx] = static_cast<MPType>(sin_input[0]);
      cos_value[nx] = static_cast<MPType>(cos_input[0]);
    }
  } else {
#pragma unroll
    for (int nx = 0; nx < VecSize; ++nx) {
      // get sin_index and cos_index
      int64_t index_wc = (index + nx) % (seq_len * num_heads * head_dim);
      int64_t pos_seq = index_wc / (num_heads * head_dim);
      MPType idx = static_cast<MPType>((index_wc % head_dim) / 2 * 2.0);
      MPType indicses =
          static_cast<MPType>(1) /
          pow(static_cast<MPType>(10000), idx * static_cast<MPType>(div_c));
      MPType value = pos_seq * indicses;
      sin_value[nx] = sin(value);
      cos_value[nx] = cos(value);
    }
  }
}

template <typename T, typename MPType, int VecSize = 2>
__device__ void VectorizedGetSinCos3D(phi::Array<const T*, 2> sin_cos_data,
                                      const int64_t* position_ids_data,
                                      bool flag_sin_cos,
                                      const int64_t batch_idx,
                                      const int64_t seq_index,
                                      const int64_t dim_index,
                                      const int64_t seq_len,
                                      const int64_t num_heads,
                                      const int64_t head_dim,
                                      MPType* out_sin,
                                      MPType* out_cos,
                                      MPType div_c) {
  MPType* sin_value = out_sin;
  MPType* cos_value = out_cos;

  if (flag_sin_cos) {
#pragma unroll
    for (int64_t nx = 0; nx < VecSize; ++nx) {
      int64_t cur_idx = dim_index + nx;
      int64_t pos_seq = seq_index;
      if (position_ids_data) {
        int64_t index_ids = batch_idx * seq_len + seq_index;
        pos_seq = position_ids_data[index_ids];
      }
      int64_t index_sc = pos_seq * head_dim + cur_idx;
      const T* sin_input = sin_cos_data[0] + index_sc;
      const T* cos_input = sin_cos_data[1] + index_sc;

      sin_value[nx] = static_cast<MPType>(sin_input[0]);
      cos_value[nx] = static_cast<MPType>(cos_input[0]);
    }

  } else {
#pragma unroll
    for (int nx = 0; nx < VecSize; ++nx) {
      // get sin_index and cos_index
      int64_t cur_idx = dim_index + nx;
      int64_t pos_seq = seq_index;
      MPType idx = static_cast<MPType>(cur_idx / 2 * 2.0);
      MPType indicses =
          static_cast<MPType>(1) /
          pow(static_cast<MPType>(10000), idx * static_cast<MPType>(div_c));
      MPType value = pos_seq * indicses;
      sin_value[nx] = sin(value);
      cos_value[nx] = cos(value);
    }
  }
}

template <typename T, typename MPType, int NInputs, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateEveryTwoKernel(
    phi::Array<const T*, NInputs> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    phi::Array<T*, NInputs> outs_data,
    int num_inputs,
    MPType div_c) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  for (; index < size; index += stride) {
    VectorizedGetSinCos(sin_cos_data,
                        position_ids_data,
                        flag_sin_cos,
                        index,
                        seq_len,
                        num_heads,
                        head_dim,
                        sin_value,
                        cos_value,
                        div_c);
    // sin_value[0] = static_cast<MPType>(1.0);
    // sin_value[1] = static_cast<MPType>(1.0);
    // cos_value[0] = static_cast<MPType>(1.0);
    // cos_value[1] = static_cast<MPType>(1.0);

#pragma unroll
    for (int iter = 0; iter < NInputs; iter++) {
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
          result[pr_index] =
              cos_value[pr_index] * p0 - sin_value[pr_index] * p1;
          result[ls_index] =
              sin_value[ls_index] * p0 + cos_value[ls_index] * p1;

        } else if (sign == -1) {
          result[pr_index] =
              cos_value[pr_index] * p0 + sin_value[ls_index] * p1;
          result[ls_index] =
              cos_value[ls_index] * p1 - sin_value[pr_index] * p0;
        }

        store[pr_index] = static_cast<T>(result[pr_index]);
        store[ls_index] = static_cast<T>(result[ls_index]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}

template <typename T, typename MPType, int NInputs, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateEveryTwoKernel3D(
    phi::Array<const T*, NInputs> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t batch_stride_q,
    int64_t seq_stride_q,
    int64_t batch_stride_kv,
    int64_t seq_stride_kv,
    phi::Array<T*, NInputs> outs_data,
    int num_inputs,
    MPType div_c) {
  // may modify
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  // ===========

  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;
  int64_t batch_index = blockIdx.x;
  int64_t seq_index = blockIdx.y;

  // sin_value[0] = static_cast<MPType>(1.0);
  // sin_value[1] = static_cast<MPType>(1.0);
  // cos_value[0] = static_cast<MPType>(1.0);
  // cos_value[1] = static_cast<MPType>(1.0);

  for (int64_t dim_index = threadIdx.x; dim_index * VecSize < head_dim;
       dim_index += blockDim.x) {
    int64_t index = dim_index * VecSize;
    VectorizedGetSinCos3D(sin_cos_data,
                          position_ids_data,
                          flag_sin_cos,
                          batch_index,
                          seq_index,
                          index,
                          seq_len,
                          num_heads,
                          head_dim,
                          sin_value,
                          cos_value,
                          div_c);

    for (int64_t head_index = threadIdx.y; head_index < num_heads;
         head_index += blockDim.y) {
      for (int iter = 0; iter < NInputs; iter++) {
        if (iter >= num_inputs) break;

        if (iter > 0 && head_index >= num_kv_heads) {
          break;
        }

        int64_t block_offset = 0;
        if (iter == 0) {
          block_offset =
              batch_index * batch_stride_q + seq_index * seq_stride_q;
        } else {
          block_offset =
              batch_index * batch_stride_kv + seq_index * seq_stride_kv;
        }

        const T* input =
            ins_data[iter] + block_offset + head_index * head_dim + index;
        VecType* out = reinterpret_cast<VecType*>(
            outs_data[iter] + block_offset + head_index * head_dim + index);

        for (int nx = 0; nx < kVectorsPerThread; ++nx) {
          int pr_index = nx * 2;
          int ls_index = pr_index + 1;

          MPType p0 = static_cast<MPType>(input[pr_index]);
          MPType p1 = static_cast<MPType>(input[ls_index]);

          if (sign == 1) {
            result[pr_index] =
                cos_value[pr_index] * p0 - sin_value[pr_index] * p1;
            result[ls_index] =
                sin_value[ls_index] * p0 + cos_value[ls_index] * p1;
          } else if (sign == -1) {
            result[pr_index] =
                cos_value[pr_index] * p0 + sin_value[ls_index] * p1;
            result[ls_index] =
                cos_value[ls_index] * p1 - sin_value[pr_index] * p0;
          }

          store[pr_index] = static_cast<T>(result[pr_index]);
          store[ls_index] = static_cast<T>(result[ls_index]);
        }

        out[0] = *(reinterpret_cast<VecType*>(store));
      }
    }
  }
}

template <typename T, typename MPType, int NInputs, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateHalfKernel3D(
    phi::Array<const T*, NInputs> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t batch_stride_q,
    int64_t seq_stride_q,
    int64_t batch_stride_kv,
    int64_t seq_stride_kv,
    phi::Array<T*, NInputs> outs_data,
    int num_inputs,
    MPType div_c) {
  using VecType = phi::AlignedVector<T, VecSize>;
  int64_t batch_index = blockIdx.x;
  int64_t seq_index = blockIdx.y;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];

  // sin_value[0] = static_cast<MPType>(1.0);
  // sin_value[1] = static_cast<MPType>(1.0);
  // cos_value[0] = static_cast<MPType>(1.0);
  // cos_value[1] = static_cast<MPType>(1.0);

#pragma unroll
  for (int64_t dim_index = threadIdx.x; dim_index * VecSize < head_dim;
       dim_index += blockDim.x) {
    int64_t index = dim_index * VecSize;

    VectorizedGetSinCos3D(sin_cos_data,
                          position_ids_data,
                          flag_sin_cos,
                          batch_index,
                          seq_index,
                          index,
                          seq_len,
                          num_heads,
                          head_dim,
                          sin_value,
                          cos_value,
                          div_c);

#pragma unroll
    for (int64_t head_index = threadIdx.y; head_index < num_heads;
         head_index += blockDim.y) {
      // use rotate_half mode
      int stride_r = head_dim / 2;

#pragma unroll
      for (int iter = 0; iter < NInputs; iter++) {
        if (iter >= num_inputs) break;

        if (iter > 0 && head_index >= num_kv_heads) {
          break;
        }

        int64_t block_offset = 0;
        if (iter == 0) {
          block_offset =
              batch_index * batch_stride_q + seq_index * seq_stride_q;
        } else {
          block_offset =
              batch_index * batch_stride_kv + seq_index * seq_stride_kv;
        }

        // get value_index and rotate_half_index
        int index_v = index;
        int index_r =
            index < stride_r ? (index + stride_r) : (index - stride_r);
        MPType sign_r =
            index < stride_r ? static_cast<MPType>(-1) : static_cast<MPType>(1);
        const T* input_v =
            ins_data[iter] + block_offset + head_index * head_dim + index_v;
        const T* input_r =
            ins_data[iter] + block_offset + head_index * head_dim + index_r;
        VecType* out = reinterpret_cast<VecType*>(
            outs_data[iter] + block_offset + head_index * head_dim + index_v);

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
  }
}

template <typename T, typename MPType, int NInputs, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateHalfKernel(
    phi::Array<const T*, NInputs> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    phi::Array<T*, NInputs> outs_data,
    int num_inputs,
    MPType div_c) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;

  for (; index < size; index += stride) {
    VectorizedGetSinCos(sin_cos_data,
                        position_ids_data,
                        flag_sin_cos,
                        index,
                        seq_len,
                        num_heads,
                        head_dim,
                        sin_value,
                        cos_value,
                        div_c);

    // use rotate_half mode
    int stride_r = head_dim / 2;
#pragma unroll
    for (int iter = 0; iter < NInputs; iter++) {
      if (iter >= num_inputs) break;
      // get value_index and rotate_half_index
      int index_v = index;
      int index_r = (index % head_dim) < stride_r ? (index + stride_r)
                                                  : (index - stride_r);
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
}

template <typename T, typename MPType, int NInputs, int VecSize = 2>
__global__ void VectorizedFusedRopeKernel(
    phi::Array<const T*, NInputs> ins_data,
    phi::Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t batch_stride_q,
    int64_t seq_stride_q,
    int64_t batch_stride_kv,
    int64_t seq_stride_kv,
    phi::Array<T*, NInputs> outs_data,
    MPType div_c,
    bool use_neox_rotary_style,
    bool flag_sin_cos,
    int sign,
    int num_inputs) {
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  int64_t batch_index = blockIdx.x;
  int64_t seq_index = blockIdx.y;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];

#pragma unroll
  for (int64_t dim_index = threadIdx.x; dim_index * VecSize < head_dim;
       dim_index += blockDim.x) {
    int64_t index = dim_index * VecSize;
    VectorizedGetSinCos3D(sin_cos_data,
                          position_ids_data,
                          flag_sin_cos,
                          batch_index,
                          seq_index,
                          index,
                          seq_len,
                          num_heads,
                          head_dim,
                          sin_value,
                          cos_value,
                          div_c);

#pragma unroll
    for (int64_t head_index = threadIdx.y; head_index < num_heads;
         head_index += blockDim.y) {
      int stride_r = head_dim / 2;

#pragma unroll
      for (int iter = 0; iter < NInputs; iter++) {
        if (iter >= num_inputs) break;

        if (iter > 0 && head_index >= num_kv_heads) {
          break;
        }

        int64_t block_offset = 0;
        if (iter == 0) {
          block_offset =
              batch_index * batch_stride_q + seq_index * seq_stride_q;
        } else {
          block_offset =
              batch_index * batch_stride_kv + seq_index * seq_stride_kv;
        }

        VecType* out = reinterpret_cast<VecType*>(
            outs_data[iter] + block_offset + head_index * head_dim + index);
        if (use_neox_rotary_style) {
          const T* input =
              ins_data[iter] + block_offset + head_index * head_dim + index;

#pragma unroll
          for (int nx = 0; nx < kVectorsPerThread; ++nx) {
            int pr_index = nx * 2;
            int ls_index = pr_index + 1;

            MPType p0 = static_cast<MPType>(input[pr_index]);
            MPType p1 = static_cast<MPType>(input[ls_index]);

            if (sign == 1) {
              result[pr_index] =
                  cos_value[pr_index] * p0 - sin_value[pr_index] * p1;
              result[ls_index] =
                  sin_value[ls_index] * p0 + cos_value[ls_index] * p1;
            } else if (sign == -1) {
              result[pr_index] =
                  cos_value[pr_index] * p0 + sin_value[ls_index] * p1;
              result[ls_index] =
                  cos_value[ls_index] * p1 - sin_value[pr_index] * p0;
            }

            store[pr_index] = static_cast<T>(result[pr_index]);
            store[ls_index] = static_cast<T>(result[ls_index]);
          }
        } else {
          // rotate_half mode
          // get value_index and rotate_half_index
          int index_v = index;
          int index_r =
              index < stride_r ? (index + stride_r) : (index - stride_r);
          MPType sign_r = index < stride_r ? static_cast<MPType>(-1)
                                           : static_cast<MPType>(1);
          const T* input_v =
              ins_data[iter] + block_offset + head_index * head_dim + index_v;
          const T* input_r =
              ins_data[iter] + block_offset + head_index * head_dim + index_r;

#pragma unroll
          for (int nx = 0; nx < VecSize; ++nx) {
            MPType p0 = static_cast<MPType>(input_v[nx]);
            MPType p1 = static_cast<MPType>(input_r[nx]);
            result[nx] =
                cos_value[nx] * p0 + sign * sign_r * sin_value[nx] * p1;
            store[nx] = static_cast<T>(result[nx]);
          }
        }

        out[0] = *(reinterpret_cast<VecType*>(store));
      }
    }
  }
}

}  // namespace fusion
}  // namespace phi
