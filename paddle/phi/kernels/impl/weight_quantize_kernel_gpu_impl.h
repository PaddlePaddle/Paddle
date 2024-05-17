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

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
namespace phi {

__global__ void weight_permute_kernel_wint8(const int8_t* input_data_dev,
                                            int8_t* output_data_dev,
                                            int numel,
                                            int total_k,
                                            int total_n) {
  for (int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
       linear_idx < numel;
       linear_idx += blockDim.x * gridDim.x) {
    int k_id = linear_idx / total_n;
    int n_id = linear_idx % total_n;
    constexpr int k_permute_const = 8;
    int k_mod_16 = k_id % 16;
    int temp_k_expr_1 = k_mod_16 - k_mod_16 / 8 * 8;
    int temp_k_expr_2 = k_mod_16 / 8;
    int permute_kk = temp_k_expr_1 + temp_k_expr_2 +
                     (temp_k_expr_2 + 1) % 2 * k_mod_16 * 2 / 2 +
                     temp_k_expr_1 * temp_k_expr_2 + k_id / 16 * 16;
    int permute_index = permute_kk % 64 + permute_kk / 64 * 128 +
                        64 * (n_id % 2) + total_k * 2 * (n_id / 2);
    uint8_t shift_quant_weight = static_cast<uint8_t>(
        static_cast<int32_t>(input_data_dev[linear_idx]) + 128);
    output_data_dev[permute_index] =
        *reinterpret_cast<int8_t*>(&shift_quant_weight);
  }
}

/*
For SM70 volta arch, weightonly int8 dequantize invoked in load global memory.
So it only need interleave in K-dimension
K_index: 0 1 2 3 -> 0 2 1 3
*/
__global__ void weight_interleave_add_bias_kernel_wint8(
    const int8_t* input_data_dev,
    int8_t* output_data_dev,
    int numel,
    int total_k,
    int total_n) {
  for (int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
       linear_idx < numel;
       linear_idx += blockDim.x * gridDim.x) {
    int k_id = linear_idx / total_n;
    int n_id = linear_idx % total_n;
    constexpr int n_interleaved_factor = 4;
    int n_interleave_group_id = n_id / n_interleaved_factor;
    int n_interleave_id = n_id % n_interleaved_factor;
    if (n_interleave_id == 1 || n_interleave_id == 2) {
      /*
      0001 xor 0011 -> 0010
      0010 xor 0011 -> 0001
      */
      n_interleave_id ^= 3;
    }
    const int new_n_id =
        n_interleave_group_id * n_interleaved_factor + n_interleave_id;
    const int interleave_idx = k_id * total_n + new_n_id;

    uint8_t shift_quant_weight = static_cast<uint8_t>(
        static_cast<int32_t>(input_data_dev[linear_idx]) + 128);
    output_data_dev[interleave_idx] =
        *reinterpret_cast<int8_t*>(&shift_quant_weight);
  }
}

template <typename GPUContext>
void weight_permute_gpu(const GPUContext& dev_ctx,
                        int8_t* input_data,
                        int8_t* output_data,
                        const std::vector<int>& shape,
                        const int32_t arch) {
  auto total_k = shape[0];
  auto total_n = shape[1];
  auto numel = total_k * total_n;
  auto gpu_config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, 1);
  int grid_size = gpu_config.GetGridSize();
  int block_size = gpu_config.GetBlockSize();
  if ((arch == 80) || (arch == 86) || (arch == 75)) {
    weight_permute_kernel_wint8<<<grid_size, block_size>>>(
        input_data, output_data, numel, total_k, total_n);
  } else if (arch == 70) {
    weight_interleave_add_bias_kernel_wint8<<<grid_size, block_size>>>(
        input_data, output_data, numel, total_k, total_n);
  }
}

template <typename T, int VectorSize = 8, typename ScaleT>
__global__ void per_channel_quant_gpu(const T* weight_data,
                                      int8_t* quanted_weight_data,
                                      ScaleT* scale_data,
                                      int total_k,
                                      int total_vec_n) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < total_vec_n) {
    const int4* vec_weight_data_ptr =
        reinterpret_cast<const int4*>(weight_data);
    int2* vec_quanted_weight_data =
        reinterpret_cast<int2*>(quanted_weight_data);
    phi::AlignedVector<float, VectorSize> abs_max;
#pragma unroll
    for (int i = 0; i < VectorSize; ++i) {
      abs_max[i] = static_cast<float>(0.0f);
    }
#pragma unroll
    for (int k = 0; k < total_k; ++k) {
      int linear_index = k * total_vec_n + n;
      phi::AlignedVector<T, VectorSize> weight;
      *reinterpret_cast<int4*>(&weight) = vec_weight_data_ptr[linear_index];
#pragma unroll
      for (int i = 0; i < VectorSize; ++i) {
        abs_max[i] = fmaxf((abs_max[i]), fabsf((weight[i])));
      }
    }
    phi::AlignedVector<ScaleT, VectorSize> scale;
#pragma unroll
    for (int i = 0; i < VectorSize; ++i) {
      scale[i] = static_cast<ScaleT>(abs_max[i] / static_cast<float>(127.0f));
    }
    *reinterpret_cast<float4*>(scale_data + VectorSize * n) =
        *reinterpret_cast<float4*>(&scale);

    for (int k = 0; k < total_k; ++k) {
      phi::AlignedVector<int8_t, VectorSize> quanted_weight;
      int linear_index = k * total_vec_n + n;
      phi::AlignedVector<T, VectorSize> weight;
      *reinterpret_cast<int4*>(&weight) =
          *reinterpret_cast<const int4*>(vec_weight_data_ptr + linear_index);
#pragma unroll
      for (int i = 0; i < VectorSize; ++i) {
        float scaled_weight =
            (static_cast<float>(weight[i]) / static_cast<float>(abs_max[i])) *
            static_cast<float>(127.0);
        int8_t clipped_weight = static_cast<int8_t>(
            lroundf(fmaxf(-127.0f, fminf(127.0f, scaled_weight))));
        quanted_weight[i] = clipped_weight;
      }
      *reinterpret_cast<int2*>(vec_quanted_weight_data + linear_index) =
          *reinterpret_cast<int2*>(&quanted_weight);
    }
  }
}
template <typename T, typename GPUContext, typename ScaleT>
void weight_quant_gpu(const GPUContext& dev_ctx,
                      const T* weight_data,
                      int8_t* quanted_weight_data,
                      ScaleT* scale_data,
                      const std::vector<int>& shape) {
  int total_k = shape[0];
  int total_n = shape[1];
  int numel = total_k * total_n;
  constexpr int kWarpSize = 32;
  constexpr int kBlockSize = 64;
  constexpr int kWarpNum = kBlockSize / kWarpSize;
  constexpr int kVectorSize = 128 / sizeof(T) / 8;
  PADDLE_ENFORCE_EQ(total_n % kVectorSize,
                    0,
                    phi::errors::PreconditionNotMet(
                        "Currently, weight_quant_gpu kernel only support n "
                        "with multiple of %d, please use",
                        kVectorSize));
  int vec_total_n = total_n / kVectorSize;
  int kGridSize =
      max((vec_total_n + kBlockSize - 1) / kBlockSize, static_cast<int>(1));
  per_channel_quant_gpu<T, kVectorSize><<<kGridSize, kBlockSize>>>(
      weight_data, quanted_weight_data, scale_data, total_k, vec_total_n);
}

}  // namespace phi
