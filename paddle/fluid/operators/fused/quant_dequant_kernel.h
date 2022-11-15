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

#include <vector>
#include "paddle/fluid/operators/fake_quantize_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__forceinline__ __device__ int8_t quant_helper(const T input,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  float quant_value = max_bound * inverse(scale) * static_cast<float>(input);
  if (round_type == 0) {
    quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename T>
__global__ void quantize_kernel(const T* input,
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

template <typename T>
void quantize_kernel_launcher(const T* input,
                              int8_t* output,
                              const float scale,
                              const int m,
                              const int n,
                              const int round_type,
                              const float max_bound,
                              const float min_bound,
                              gpuStream_t stream) {
  // TODO(minghaoBD): optimize the kennel launch times when m==1 or n==1
  dim3 grid((n + 31) / 32, (m + 31) / 32);
  dim3 block(32, 32);

  quantize_kernel<<<grid, block, 0, stream>>>(input,
                                              (char4*)output,  // NOLINT
                                              scale,
                                              m,
                                              n,
                                              round_type,
                                              max_bound,
                                              min_bound);
}

// dequantize using weight scales and input scales
template <typename T>
__global__ void dequantize_kernel(T* output,
                                  const int32_t* input,
                                  const int m,  // hidden
                                  const int n,  // batch size
                                  const float quant_in_scale,
                                  const float* dequant_out_scale_data,
                                  const int quant_out_scale_offset) {
  int m_id = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int n_id = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    float out_scale = dequant_out_scale_data[quant_out_scale_offset + m_id];
    output[n_id * m + m_id] =
        static_cast<T>(static_cast<float>(input[n_id * m + m_id]) *
                       quant_in_scale / out_scale);
  }
}

template <typename T>
void dequantize_kernel_launcher(const int32_t* input,
                                T* output,
                                const int batch_size,    // m
                                const int hidden_units,  // n
                                gpuStream_t stream,
                                const float quant_in_scale,
                                const float* dequant_out_scale_data,
                                const int quant_out_scale_offset) {
  dim3 grid((hidden_units + 31) / 32, (batch_size + 31) / 32);
  dim3 block(32, 32);

  dequantize_kernel<<<grid, block, 0, stream>>>(output,
                                                input,
                                                hidden_units,
                                                batch_size,
                                                quant_in_scale,
                                                dequant_out_scale_data,
                                                quant_out_scale_offset);
}

}  // namespace operators
}  // namespace paddle
