// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/fast_divmod.h"
#include "paddle/phi/kernels/funcs/segmented_array.h"
#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"
#include "paddle/phi/kernels/primitive/datamover_primitives.h"

namespace phi {
namespace fusion {

using FastDivMod = phi::funcs::FastDivMod<int64_t>;

template <typename ArrayT>
__device__ void BlockLoad(ArrayT input_array,
                          __nv_bfloat16 x[8][4],
                          size_t K,
                          size_t block_y,
                          size_t block_x) {
  const __nv_bfloat16* input =
      reinterpret_cast<const __nv_bfloat16*>(input_array.data[blockIdx.z]);

  for (size_t i = 0; i < 8; i++) {
    size_t idx_m = block_y * 128 + threadIdx.y + i * 16;
    size_t idx_k = block_x * 128 + threadIdx.x * 4;
    size_t idx = idx_m * K + idx_k;

    using LoadT = phi::kps::details::VectorType<__nv_bfloat16, 4>;
    LoadT data = *reinterpret_cast<const LoadT*>(input + idx);
    for (int j = 0; j < 4; j++) {
      x[i][j] = data.val[j];
    }
  }
}

template <int Width = 32>
__device__ __nv_bfloat16 WarpReduceMax(__nv_bfloat16 x) {
  constexpr unsigned mask = (uint64_t(1) << Width) - 1;
  for (int offset = Width / 2; offset > 0; offset /= 2) {
    __nv_bfloat16 t = __shfl_down_sync(mask, x, offset);
    x = BF16_MAX(x, t);
  }
  return x;
}

template <typename OutT>
__device__ float BlockReduceScale(__nv_bfloat16 x[8][4]) {
  // [(8), 16, 32, (4)] => [16, 32]
  __nv_bfloat16 local_max;
  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 4; j++) {
      __nv_bfloat16 t = BF16_ABS(x[i][j]);
      local_max = (i == 0 && j == 0) ? t : BF16_MAX(local_max, t);
    }
  }

  // [16, (32)] => [16]
  __nv_bfloat16 warp_max = WarpReduceMax(local_max);

  // [(16)] => [1]
  __shared__ __nv_bfloat16 block_max[16];
  __shared__ float block_scale;
  if (threadIdx.x == 0) {
    block_max[threadIdx.y] = warp_max;
  }
  __syncthreads();
  if (threadIdx.y == 0 && threadIdx.x < 16) {
    warp_max = WarpReduceMax<16>(block_max[threadIdx.x]);
    if (threadIdx.x == 0) {
      block_scale = ComputeScale<__nv_bfloat16, OutT>(warp_max, 0.0f);
    }
  }
  __syncthreads();

  return block_scale;
}

template <typename OutT, typename ArrayT>
__global__ void __launch_bounds__(512)
    FusedStackQuantGPUKernel(ArrayT input_array,
                             OutT* __restrict__ out,
                             float* __restrict__ scale,
                             size_t M,
                             size_t K,
                             FastDivMod K_div_128) {
  size_t block_y = K_div_128.Div(blockIdx.x);
  size_t block_x = blockIdx.x - block_y * (K / 128);

  // Load 128x128 elements from X
  __nv_bfloat16 x[8][4];
  BlockLoad(input_array, x, K, block_y, block_x);

  // Find the scale of all elements
  float block_scale = BlockReduceScale<OutT>(x);

  // Compute scale and store back
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t idx_n = blockIdx.z;
    size_t idx_m = block_y;
    size_t idx_k = block_x;
    size_t idx = (idx_n * (M / 128) + idx_m) * (K / 128) + idx_k;
    scale[idx] = __frcp_rn(block_scale);
  }

  // Scale X and store to out
  for (uint32_t i = 0; i < 8; i++) {
    size_t idx_n = blockIdx.z;
    size_t idx_m = block_y * 128 + threadIdx.y + i * 16;
    size_t idx_k = block_x * 128 + threadIdx.x * 4;
    size_t idx = (idx_n * M + idx_m) * K + idx_k;

    using StoreT = phi::kps::details::VectorType<OutT, 4>;
    StoreT data;
    for (uint32_t j = 0; j < 4; j++) {
      float x_fp32 = static_cast<float>(x[i][j]);
      float output_scaled = x_fp32 * block_scale;
      data.val[j] = static_cast<OutT>(output_scaled);
    }
    *reinterpret_cast<StoreT*>(out + idx) = data;
  }
}

template <typename OutT, typename ArrayT>
__global__ void __launch_bounds__(512)
    FusedStackTransposeQuantGPUKernel(ArrayT input_array,
                                      OutT* __restrict__ out,
                                      float* __restrict__ scale,
                                      size_t M,
                                      size_t K,
                                      FastDivMod K_div_128) {
  size_t block_y = K_div_128.Div(blockIdx.x);
  size_t block_x = blockIdx.x - block_y * (K / 128);

  // Load 128x128 elements from X
  __nv_bfloat16 x[8][4];
  BlockLoad(input_array, x, K, block_y, block_x);

  // Find the scale of all elements
  float block_scale = BlockReduceScale<OutT>(x);

  // Compute scale and store back
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t idx_n = blockIdx.z;
    size_t idx_k = block_x;
    size_t idx_m = block_y;
    size_t idx = (idx_n * (K / 128) + idx_k) * (M / 128) + idx_m;
    scale[idx] = __frcp_rn(block_scale);
  }

  // Scale X and transpose in shared memory
  __shared__ OutT shm[128][129];
  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 4; j++) {
      float x_fp32 = static_cast<float>(x[i][j]);
      float output_scaled = x_fp32 * block_scale;
      shm[threadIdx.x * 4 + j][i * 16 + threadIdx.y] =
          static_cast<OutT>(output_scaled);
    }
  }
  __syncthreads();

  // Store X back to out
  for (uint32_t i = 0; i < 8; i++) {
    size_t idx_n = blockIdx.z;
    size_t idx_k = block_x * 128 + threadIdx.y + i * 16;
    size_t idx_m = block_y * 128 + threadIdx.x * 4;
    size_t idx = (idx_n * K + idx_k) * M + idx_m;

    using StoreT = phi::kps::details::VectorType<OutT, 4>;
    StoreT data;
    for (uint32_t j = 0; j < 4; j++) {
      data.val[j] = shm[i * 16 + threadIdx.y][threadIdx.x * 4 + j];
    }
    *reinterpret_cast<StoreT*>(out + idx) = data;
  }
}

template <typename T, typename Context>
void FusedStackTransposeQuantImpl(const Context& dev_ctx,
                                  const std::vector<const DenseTensor*>& x,
                                  bool transpose,
                                  DenseTensor* out,
                                  DenseTensor* scale) {
  int N = static_cast<int>(x.size());

  // zero sized tensor case
  if (x[0]->numel() == 0) {
    dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(out);
    dev_ctx.template Alloc<float>(scale);
    return;
  }

  int64_t M = x[0]->dims()[0];
  int64_t K = x[0]->dims()[1];

  dim3 grid((M / 128) * (K / 128), 1, N);
  dim3 block(32, 16);
  auto* out_data = dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(out);
  auto* scale_data = dev_ctx.template Alloc<float>(scale);
  FastDivMod K_div_128(K / 128);

  switch (funcs::CalcArraySize(N)) {
    SEGMENTED_ARRAY_KERNEL_HELPER({
      funcs::ConstPointerArraySetter<Context, T, kArraySize> setter(dev_ctx, x);
      if (transpose) {
        FusedStackTransposeQuantGPUKernel<phi::dtype::float8_e4m3fn>
            <<<grid, block, 0, dev_ctx.stream()>>>(
                setter.array, out_data, scale_data, M, K, K_div_128);
      } else {
        FusedStackQuantGPUKernel<phi::dtype::float8_e4m3fn>
            <<<grid, block, 0, dev_ctx.stream()>>>(
                setter.array, out_data, scale_data, M, K, K_div_128);
      }
    });
  }
}

template <typename T, typename Context>
void FusedStackQuantKernel(const Context& dev_ctx,
                           const std::vector<const DenseTensor*>& x,
                           DenseTensor* out,
                           DenseTensor* scale) {
  FusedStackTransposeQuantImpl<T>(dev_ctx, x, false, out, scale);
}

template <typename T, typename Context>
void FusedStackTransposeQuantKernel(const Context& dev_ctx,
                                    const std::vector<const DenseTensor*>& x,
                                    DenseTensor* out,
                                    DenseTensor* scale) {
  FusedStackTransposeQuantImpl<T>(dev_ctx, x, true, out, scale);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_stack_quant,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedStackQuantKernel,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT8_E4M3FN);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
}

PD_REGISTER_KERNEL(fused_stack_transpose_quant,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedStackTransposeQuantKernel,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT8_E4M3FN);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
}
