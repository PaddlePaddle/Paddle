/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/phi/kernels/funcs/aligned_vector.h"

#define VecBytes 16

namespace phi {
namespace funcs {
namespace sparse {

/**
 * brief: scatter add
 * input: the inputs
 * unique_value: refer to UpdateIndexKernel notes
 * out_index: the output feature index
 * non_zero_num: the number of output features
 * rulebook_len: the length of rulebook
 * channels: the output channel size
 * out: the outputs
 **/
template <typename T, int VecSize>
__global__ void ScatterKernel(const T* input,
                              const int* unique_value,
                              const int* out_index,
                              const int non_zero_num,
                              const int rulebook_len,
                              const int channels,
                              T* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int vec_channels = channels / VecSize;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  for (int i = tid; i < non_zero_num * vec_channels;
       i += gridDim.x * blockDim.x) {
    int indices_i = i / vec_channels;
    int channels_i = i - indices_i * vec_channels;

    int start = unique_value[indices_i];
    int end = indices_i == non_zero_num - 1 ? rulebook_len
                                            : unique_value[indices_i + 1];
    // max(end-start) = kernel_size
    StoreT sums = {static_cast<T>(0)};
    for (int j = start; j < end; j++) {
      const int out_feature_i = out_index[j];
      LoadT vec_in;
      phi::Load<T, VecSize>(
          input + out_feature_i * channels + channels_i * VecSize, &vec_in);
#pragma unroll
      for (int k = 0; k < VecSize; k++) {
        sums[k] += vec_in[k];
      }
    }
    phi::Store<T, VecSize>(sums,
                           out + indices_i * channels + channels_i * VecSize);
  }
}

template <typename T, typename IndexT = int, int VecSize>
__global__ void ScatterCUDAKernel(const T* params,
                                  const IndexT* indices,
                                  T* output,
                                  size_t index_size,
                                  size_t slice_size,
                                  bool overwrite) {
  const size_t vec_slice_size = slice_size / VecSize;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  CUDA_KERNEL_LOOP_TYPE(i, index_size * vec_slice_size, int64_t) {
    int64_t indices_i = i / vec_slice_size;
    int64_t slice_i =
        i - indices_i * vec_slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];

    int64_t out_i = scatter_i * slice_size + slice_i * VecSize;
    LoadT vec_params, vec_out;
    phi::Load<T, VecSize>(params + i * VecSize, &vec_params);
    phi::Load<T, VecSize>(output + out_i, &vec_out);
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      vec_out[j] += vec_params[j];
    }
    phi::Store<T, VecSize>(vec_out, output + out_i);
    // output[out_i] += params[i];
  }
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
