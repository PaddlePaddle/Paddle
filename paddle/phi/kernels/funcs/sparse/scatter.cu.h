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
template <typename T>
__global__ void ScatterKernel(const T* input,
                              const int* unique_value,
                              const int* out_index,
                              const int non_zero_num,
                              const int rulebook_len,
                              const int channels,
                              T* out,
                              const bool subm = false) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num * channels; i += gridDim.x * blockDim.x) {
    int indices_i = i / channels;
    int channels_i = i - indices_i * channels;

    int start = unique_value[indices_i];
    int end = indices_i == non_zero_num - 1 ? rulebook_len
                                            : unique_value[indices_i + 1];
    // max(end-start) = kernel_size
    T sum = static_cast<T>(0);
    if (subm) {
      sum = out[indices_i * channels + channels_i];
    }
    for (int j = start; j < end; j++) {
      const int out_feature_i = out_index[j];
      sum += input[out_feature_i * channels + channels_i];
    }
    out[indices_i * channels + channels_i] = sum;
  }
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
