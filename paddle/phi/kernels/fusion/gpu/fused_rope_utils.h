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

template <typename T, typename MPType, int VecSize = 2>
__global__ void VectorizedFusedRopeKernel(phi::Array<const T*, 3> ins_data,
                                          phi::Array<const T*, 2> sin_cos_data,
                                          bool flag_sin_cos,
                                          int sign,
                                          int batch_size,
                                          int seq_len,
                                          int num_heads,
                                          int head_dim,
                                          phi::Array<T*, 3> outs_data,
                                          int num_inputs,
                                          MPType div_c) {
  int index = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int stride = gridDim.x * blockDim.x * VecSize;
  int size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = phi::AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  for (; index < size; index += stride) {
    if (flag_sin_cos) {
#pragma unroll
      for (int nx = 0; nx < VecSize; ++nx) {
        int index_wc = (index + nx) % (seq_len * num_heads * head_dim);
        int pos_seq = index_wc / (num_heads * head_dim);
        int pos_head = index_wc % head_dim;
        int index_sc = pos_seq * head_dim + pos_head;
        const T* sin_input = sin_cos_data[0] + index_sc;
        const T* cos_input = sin_cos_data[1] + index_sc;

        sin_value[nx] = static_cast<MPType>(sin_input[0]);
        cos_value[nx] = static_cast<MPType>(cos_input[0]);
      }
    } else {
#pragma unroll
      for (int nx = 0; nx < VecSize; ++nx) {
        // get sin_index and cos_index
        int index_wc = (index + nx) % (seq_len * num_heads * head_dim);
        int pos_seq = index_wc / (num_heads * head_dim);
        MPType idx = static_cast<MPType>((index_wc % head_dim) / 2 * 2.0);
        MPType indicses =
            static_cast<MPType>(1) /
            pow(static_cast<MPType>(10000), idx * static_cast<MPType>(div_c));
        MPType value = pos_seq * indicses;
        sin_value[nx] = sin(value);
        cos_value[nx] = cos(value);
      }
    }

#pragma unroll
    for (int iter = 0; iter < 3; iter++) {
      if (iter > num_inputs) break;
      const T* input = ins_data[iter] + index;
      VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
      for (int nx = 0; nx < kVectorsPerThread; ++nx) {
        int pr_index = nx * 2;
        int ls_index = pr_index + 1;

        MPType p0 = static_cast<MPType>(input[pr_index]);
        MPType p1 = static_cast<MPType>(input[ls_index]);

        result[pr_index] =
            cos_value[pr_index] * p0 - sign * sin_value[ls_index] * p1;
        result[ls_index] =
            cos_value[ls_index] * p1 + sign * sin_value[pr_index] * p0;

        store[pr_index] = static_cast<T>(result[pr_index]);
        store[ls_index] = static_cast<T>(result[ls_index]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}

}  // namespace fusion
}  // namespace phi
