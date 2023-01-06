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
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

namespace paddle {
namespace operators {

using phi::backends::gpu::GpuLaunchConfig;

constexpr int DequantKernelVecSize = 4;

template <typename T>
__forceinline__ __device__ int8_t quant_helper(const T input,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);

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
                                const T* quant_in_scale_gpu,
                                const int m,
                                const int n,
                                const int round_type,
                                const float max_bound,
                                const float min_bound) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  auto quant_in_scale = quant_in_scale_gpu
                            ? (1.0f / static_cast<float>(quant_in_scale_gpu[0]))
                            : scale;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    char4 tmp;
    tmp.x = quant_helper(input[m_id * n + n_id],
                         quant_in_scale,
                         round_type,
                         max_bound,
                         min_bound);
    tmp.y = quant_helper(input[m_id * n + n_id + 1],
                         quant_in_scale,
                         round_type,
                         max_bound,
                         min_bound);
    tmp.z = quant_helper(input[m_id * n + n_id + 2],
                         quant_in_scale,
                         round_type,
                         max_bound,
                         min_bound);
    tmp.w = quant_helper(input[m_id * n + n_id + 3],
                         quant_in_scale,
                         round_type,
                         max_bound,
                         min_bound);
    output[(m_id * n + n_id) >> 2] = tmp;
  }
}

template <typename T>
void quantize_kernel_launcher(const T* input,
                              int8_t* output,
                              const float scale,
                              const T* quant_in_scale_gpu,
                              const int m,
                              const int n,
                              const int round_type,
                              const float max_bound,
                              const float min_bound,
                              gpuStream_t stream) {
  // TODO(minghaoBD): optimize the kennel launch times when m==1 or n==1
  dim3 grid((n >> 2 + 31) / 32, (m + 31) / 32);
  dim3 block(32, 32);

  quantize_kernel<<<grid, block, 0, stream>>>(input,
                                              (char4*)output,  // NOLINT
                                              scale,
                                              quant_in_scale_gpu,
                                              m,
                                              n,
                                              round_type,
                                              max_bound,
                                              min_bound);
}

template <typename T, int VecSize>
__global__ void dequantize_kernel(T* output,
                                  const int32_t* input,
                                  const int m,  // batch size
                                  const int n,  // hidden
                                  const float quant_in_scale,
                                  const T* quant_in_scale_gpu,
                                  const float* dequant_out_scale_data) {
  int numel = m * n;
  int stride = blockDim.x * gridDim.x * VecSize;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  int col_id = idx % n;

  phi::AlignedVector<int32_t, VecSize> in_vec;
  phi::AlignedVector<float, VecSize> out_scale_vec;
  phi::AlignedVector<T, VecSize> out_vec;

  float real_quant_in_scale = 0;
  if (quant_in_scale_gpu) {
    real_quant_in_scale = static_cast<float>(quant_in_scale_gpu[0]) / 127.0f;
  }

  for (; idx < numel; idx += stride) {
    phi::Load<int32_t, VecSize>(input + idx, &in_vec);
    phi::Load<float, VecSize>(dequant_out_scale_data + col_id, &out_scale_vec);

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      if (!quant_in_scale_gpu) {
        out_vec[i] =
            static_cast<T>(static_cast<float>(in_vec[i]) * out_scale_vec[i]);
      } else {
        out_vec[i] = static_cast<T>(static_cast<float>(in_vec[i]) *
                                    real_quant_in_scale * out_scale_vec[i]);
      }
    }

    phi::Store<T, VecSize>(out_vec, output + idx);
  }
}

template <typename T>
void dequantize_kernel_launcher(const int32_t* input,
                                T* output,
                                const int m,  // m
                                const int n,  // n
                                gpuStream_t stream,
                                GpuLaunchConfig* gpu_config,
                                const float quant_in_scale,
                                const T* quant_in_scale_gpu,
                                const float* dequant_out_scale_data) {
  VLOG(1) << "Launch dequantize_kernel";
  dequantize_kernel<T, DequantKernelVecSize>
      <<<gpu_config->block_per_grid, gpu_config->thread_per_block, 0, stream>>>(
          output,
          input,
          m,
          n,
          quant_in_scale,
          quant_in_scale_gpu,
          dequant_out_scale_data);
}

template <typename T>
void max_kernel_launcher(const phi::GPUContext& dev_ctx,
                         const T* input,
                         T* output,
                         int num_items) {
  auto stream = dev_ctx.stream();
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(
      nullptr, temp_storage_bytes, input, output, num_items, stream);
  phi::DenseTensor tmp = phi::Empty<uint8_t, phi::GPUContext>(
      dev_ctx, {static_cast<int64_t>(temp_storage_bytes)});

  auto* temp_storage = dev_ctx.Alloc<uint8_t>(&tmp);

  cub::DeviceReduce::Max(
      temp_storage, temp_storage_bytes, input, output, num_items, stream);
}

template <>
inline void max_kernel_launcher<platform::float16>(
    const phi::GPUContext& dev_ctx,
    const platform::float16* input,
    platform::float16* output,
    int num_items) {
  auto stream = dev_ctx.stream();
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(nullptr,
                         temp_storage_bytes,
                         reinterpret_cast<const half*>(input),
                         reinterpret_cast<half*>(output),
                         num_items,
                         stream);
  phi::DenseTensor tmp = phi::Empty<uint8_t, phi::GPUContext>(
      dev_ctx, {static_cast<int64_t>(temp_storage_bytes)});

  auto* temp_storage = dev_ctx.Alloc<uint8_t>(&tmp);

  cub::DeviceReduce::Max(temp_storage,
                         temp_storage_bytes,
                         reinterpret_cast<const half*>(input),
                         reinterpret_cast<half*>(output),
                         num_items,
                         stream);
}

}  // namespace operators
}  // namespace paddle
