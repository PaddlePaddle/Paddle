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

// from
// 0 1 2 3 4 5 6 7...
// to
// 0 8 16 24 1 9 17 25...
__global__ void weight_permute_kernel_wint4(const int8_t* input_data_dev,
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
    int k_mod_8 = k_id % 8;
    int temp_k_expr_1 = k_mod_8 - k_mod_8 / 4 * 4;
    int temp_k_expr_2 = k_mod_8 / 4;
    // we need int4 index like
    // 0 8 16 24 1 9 17 25 2 10 18 26 3 11 19 27
    // 4 12 20 28 5 13 21 29 6 14 22 30 7 15 23 31
    // we can change it to
    // 0 1 16 17 8 9 24 25 2 3 18 19 10 11 26 27
    // 4 5 20 21 12 13 28 29 6 7 22 23 14 15 30 31
    // 2 int4 pack to a int8
    // 0 8 4 12 1 9 5 13 2 10 6 14 3 11 7 15
    // find index of above list
    // 0 4 8 12 2 6 10 14 1 5 9 13 3 7 11 15
    // we know int8 index is 
    // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15
    // change it to
    // 0 2 4 6 1 3 5 7 8 10 12 14 9 11 13 15
    // % 8 * 2
    // 0 4 8 12 2 6 10 14 0 4 8 12 2 6 10 14
    // add 1 for 0 4 8 12 2 6 10 14 [0 4 8 12 2 6 10 14]
    // we get 0 4 8 12 2 6 10 14 1 5 9 13 3 7 11 15
    // it change ori to 0 8 4 12...
    // finally we do some bitwise operation to change int4index
    int permute_kk = (temp_k_expr_1 + temp_k_expr_2 +
                     (temp_k_expr_2 + 1) % 2 * k_mod_8 * 2 / 2 +
                     temp_k_expr_1 * temp_k_expr_2) % 8 * 2 + 
                     (k_id % 16) / 8 + k_id / 16 * 16;
    int permute_index = permute_kk % 32 + permute_kk / 32 * 128 + 
                        32 * (n_id % 4) + total_k * 4 * (n_id / 4);
    int8_t shift_quant_weight = input_data_dev[linear_idx];
    output_data_dev[permute_index] =
        *reinterpret_cast<int8_t*>(&shift_quant_weight);
  }
  constexpr int value_per_interval_thread = 4;
  constexpr int pack_size = 2;
  for (int linear_idx = blockIdx.x * blockDim.x + threadIdx.x * value_per_interval_thread;
       linear_idx < numel;
       linear_idx += blockDim.x * gridDim.x * 4) {
    for (int pack = 0; pack < pack_size; ++pack) {
      uint8_t interval_weight_0 = static_cast<uint8_t>(
        static_cast<int32_t>(output_data_dev[linear_idx + pack]));
      uint8_t interval_weight_1 = static_cast<uint8_t>(
          static_cast<int32_t>(output_data_dev[linear_idx + pack + 2]));

      uint8_t interval_weight_0_l = static_cast<uint8_t>(((interval_weight_0 & 0x0F) + 8) & 0x0F);
      uint8_t interval_weight_0_r = static_cast<uint8_t>(((interval_weight_0 >> 4) + 8) & 0x0F);
      uint8_t interval_weight_1_l = static_cast<uint8_t>(((interval_weight_1 & 0x0F) + 8) & 0x0F);
      uint8_t interval_weight_1_r = static_cast<uint8_t>(((interval_weight_1 >> 4) + 8) & 0x0F);

      uint8_t new_interval_weight_0 = interval_weight_0_l | (interval_weight_1_l << 4);
      uint8_t new_interval_weight_1 = interval_weight_0_r | (interval_weight_1_r << 4);

      output_data_dev[linear_idx + pack] =
          *reinterpret_cast<int8_t*>(&new_interval_weight_0);
      output_data_dev[linear_idx + pack + 2] =
          *reinterpret_cast<int8_t*>(&new_interval_weight_1);
    }
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

/*
For SM70 volta arch, weightonly int8 dequantize invoked in load global memory.
So it only need interleave in K-dimension
K_index: 0 1 2 3 4 5 6 7 -> 0 2 4 6 1 3 5 7
*/
__global__ void weight_interleave_add_bias_kernel_wint4(
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
    constexpr int n_interleaved_factor = 8;
    int n_interleave_group_id = n_id / n_interleaved_factor;
    int n_interleave_id = n_id % n_interleaved_factor;

    int n_interleave_offset = n_interleave_id / 4;
    n_interleave_id = (n_interleave_id % 4) * 2 + n_interleave_offset;
    const int new_n_id =
        n_interleave_group_id * n_interleaved_factor + n_interleave_id;
    const int interleave_idx = k_id * total_n + new_n_id;

    uint8_t offseted_weight = 0;
    uint8_t shift_quant_weight = static_cast<uint8_t>(
        static_cast<int32_t>(input_data_dev[linear_idx]));
    uint8_t shift_quant_weight_low = ((shift_quant_weight & 0x0F) + 8) & 0x0F;
    uint8_t shift_quant_weight_high = ((shift_quant_weight >> 4 & 0x0F) + 8) & 0x0F;
    offseted_weight = shift_quant_weight_low | (shift_quant_weight_high << 4);
    output_data_dev[interleave_idx] =
        *reinterpret_cast<int8_t*>(&offseted_weight);
  }
}

template <typename GPUContext>
void weight_permute_gpu(const GPUContext& dev_ctx,
                        int8_t* input_data,
                        int8_t* output_data,
                        const std::vector<int>& shape,
                        const int32_t arch,
                        const std::string& algo) {
  auto total_k = shape[0];
  auto total_n = shape[1];
  printf("total_k: %d, total_n: %d", total_k, total_n);
  auto numel = total_k * total_n;
  auto gpu_config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, 1);
  int grid_size = gpu_config.GetGridSize();
  int block_size = gpu_config.GetBlockSize();
  if ((arch == 80) || (arch == 86) || (arch == 75)) {
    if (algo == "weight_only_int4") {
      total_k /= 2;
      numel /= 2;
      weight_permute_kernel_wint4<<<grid_size, block_size>>>(
        input_data, output_data, numel, total_k, total_n);
    } else {
      weight_permute_kernel_wint8<<<grid_size, block_size>>>(
        input_data, output_data, numel, total_k, total_n);
    }
  } else if (arch == 70) {
    if (algo == "weight_only_int4") {
      total_k /= 2;
      numel /= 2;
      weight_permute_kernel_wint4<<<grid_size, block_size>>>(
        input_data, output_data, numel, total_k, total_n);
    } else {
      weight_permute_kernel_wint8<<<grid_size, block_size>>>(
        input_data, output_data, numel, total_k, total_n);
    }
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

template <typename T, int VectorSize = 8, typename ScaleT>
__global__ void per_channel_quant_gpu_int4(const T* weight_data,
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
      scale[i] = static_cast<ScaleT>(abs_max[i] / static_cast<float>(7.0f));
    }
    *reinterpret_cast<float4*>(scale_data + VectorSize * n) =
        *reinterpret_cast<float4*>(&scale);

    for (int k = 0; k < total_k / 2; ++k) {
      phi::AlignedVector<int8_t, VectorSize> quanted_weight;
      for (int i = 0; i < VectorSize; ++i) {
        quanted_weight[i] = 0;
      }
      for (int packed_idx = 0; packed_idx < 2; ++packed_idx) {
        int linear_index = (k * 2 + packed_idx) * total_vec_n + n;
        phi::AlignedVector<T, VectorSize> weight;
        *reinterpret_cast<int4*>(&weight) =
            *reinterpret_cast<const int4*>(vec_weight_data_ptr + linear_index);
  #pragma unroll
        for (int i = 0; i < VectorSize; ++i) {
          float scaled_weight =
              (static_cast<float>(weight[i]) / static_cast<float>(abs_max[i])) *
              static_cast<float>(7.0);
          int8_t clipped_weight = static_cast<int8_t>(
              lroundf(fmaxf(-7.0f, fminf(7.0f, scaled_weight))));
          quanted_weight[i] |= ((clipped_weight & 0x0F) << (4 * packed_idx));
        }
      }
      int linear_index_new = k * total_vec_n + n;
      *reinterpret_cast<int2*>(vec_quanted_weight_data + linear_index_new) =
          *reinterpret_cast<int2*>(&quanted_weight);
    }
  }
}

template <typename T, typename GPUContext, typename ScaleT>
void weight_quant_gpu(const GPUContext& dev_ctx,
                      const T* weight_data,
                      int8_t* quanted_weight_data,
                      ScaleT* scale_data,
                      const std::vector<int>& shape,
                      const std::string& algo) {
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
  if (algo == "weight_only_int4") {
    per_channel_quant_gpu_int4<T, kVectorSize><<<kGridSize, kBlockSize>>>(
      weight_data, quanted_weight_data, scale_data, total_k, vec_total_n);
  } else {
    per_channel_quant_gpu<T, kVectorSize><<<kGridSize, kBlockSize>>>(
      weight_data, quanted_weight_data, scale_data, total_k, vec_total_n);
  }
}

}  // namespace phi
