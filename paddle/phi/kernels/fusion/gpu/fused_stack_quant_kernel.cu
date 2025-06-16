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
#include "paddle/phi/kernels/fusion/gpu/fused_stack_transpose_quant.h"

namespace phi {
namespace fusion {

template <typename OutT>
__global__ void __launch_bounds__(1024)
    FusedStackQuantGPUKernel(const int64_t* __restrict__ X_ptrs,
                             OutT* __restrict__ out,
                             float* __restrict__ scale,
                             size_t M,
                             size_t K,
                             FastDiv K_div_128) {
  size_t block_y = K_div_128.Div(blockIdx.x);
  size_t block_x = blockIdx.x - block_y * (K / 128);

  // Load 128x128 elements from X
  __nv_bfloat16 input[4][4];
  BlockLoad(X_ptrs, input, K, block_y, block_x);

  // Find the maximum in all elements
  __nv_bfloat16 amax = BlockReduceMax(input);

  // Compute scale and store back
  float scale_inv = ComputeScale<__nv_bfloat16, OutT>(amax, 0.0f);
  float scale_out = __frcp_rn(scale_inv);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t idx_n = blockIdx.z;
    size_t idx_m = block_y;
    size_t idx_k = block_x;
    size_t idx = (idx_n * (M / 128) + idx_m) * (K / 128) + idx_k;
    scale[idx] = scale_out;
  }

  // Scale X and store to out
  for (size_t i = 0; i < 4; i++) {
    size_t idx_n = blockIdx.z;
    size_t idx_m = block_y * 128 + threadIdx.y + i * 32;
    size_t idx_k = block_x * 128 + threadIdx.x * 4;
    size_t idx = (idx_n * M + idx_m) * K + idx_k;

    using StoreT = VecType<OutT, 4>;
    StoreT data;
    for (int j = 0; j < 4; j++) {
      float input_fp32 = static_cast<float>(input[i][j]);
      float output_scaled = input_fp32 * scale_inv;
      data[j] = static_cast<OutT>(output_scaled);
    }
    *reinterpret_cast<StoreT*>(out + idx) = data;
  }
}

template <typename T, typename Context>
void FusedStackQuantKernel(const Context& dev_ctx,
                           const std::vector<const DenseTensor*>& x,
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

  DenseTensor x_ptrs_cpu;
  x_ptrs_cpu.Resize({N});

  int64_t* cpu_data = dev_ctx.template HostAlloc<int64_t>(&x_ptrs_cpu);
  for (int64_t i = 0; i < N; i++) {
    cpu_data[i] = reinterpret_cast<int64_t>(x[i]->data<T>());
  }
  DenseTensor x_ptrs_gpu;
  int64_t* x_ptrs_gpu_data = dev_ctx.template Alloc<int64_t>(&x_ptrs_gpu);
  phi::Copy(dev_ctx, x_ptrs_cpu, dev_ctx.GetPlace(), true, &x_ptrs_gpu);

  dim3 grid((M / 128) * (K / 128), 1, N);
  dim3 block(32, 32);
  auto* out_data = dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(out);
  auto* scale_data = dev_ctx.template Alloc<float>(scale);
  FusedStackQuantGPUKernel<phi::dtype::float8_e4m3fn>
      <<<grid, block, 0, dev_ctx.stream()>>>(x_ptrs_gpu.data<int64_t>(),
                                             out_data,
                                             scale_data,
                                             M,
                                             K,
                                             FastDiv(K / 128));
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
