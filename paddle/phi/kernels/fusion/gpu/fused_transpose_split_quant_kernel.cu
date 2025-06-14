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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"

namespace phi {

template <typename T, int VecSize>
struct __align__(sizeof(T) * VecSize) VecType {
  T val[VecSize];
  __host__ __device__ inline T& operator[](size_t i) { return val[i]; }
  __host__ __device__ inline const T& operator[](size_t i) const {
    return val[i];
  }
};

template <int VecSize>
__device__ void BlockLoad(const phi::bfloat16* X,
                          __nv_bfloat16 input[4][4],
                          size_t K) {
  for (size_t i = 0; i < 4; i++) {
    size_t off_m = blockIdx.x * 128 + threadIdx.y + i * 32;
    size_t off_k = blockIdx.y * 128 + threadIdx.x * VecSize;
    size_t offset = off_m * K + off_k;

    for (size_t j = 0; j < 4; j += VecSize) {
      if (off_k + j * 32 < K) {
        size_t idx = offset + j * 32;
        using LoadT = VecType<__nv_bfloat16, VecSize>;
        LoadT data = *reinterpret_cast<const LoadT*>(X + idx);
        for (int k = 0; k < VecSize; k++) {
          input[i][j + k] = data[k];
        }
      }
    }
  }
}

__device__ void BlockColumnMax(const __nv_bfloat16 input[4][4],
                               __nv_bfloat16 amax[4],
                               __nv_bfloat16* shm) {
  // Reduce [(4), 32, 32, 4] => [32, 32, 4]
  __nv_bfloat16 warp_max[4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      __nv_bfloat16 t = BF16_ABS(input[i][j]);
      warp_max[j] = i == 0 ? t : BF16_MAX(warp_max[j], t);
    }
  }

  // Reduce [(32), 32, 4] => [32, 4]
  for (int i = 0; i < 4; i++) {
    shm[threadIdx.y * 128 + i * 32 + threadIdx.x] = warp_max[i];
  }
  __syncthreads();
  for (int offset = 16; offset > 0; offset /= 2) {
    if (threadIdx.y < offset) {
      for (int i = 0; i < 4; i++) {
        shm[threadIdx.y * 128 + i * 32 + threadIdx.x] =
            BF16_MAX(shm[threadIdx.y * 128 + i * 32 + threadIdx.x],
                     shm[(threadIdx.y + offset) * 128 + i * 32 + threadIdx.x]);
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; i++) {
    amax[i] = shm[i * 32 + threadIdx.x];
  }
  __syncthreads();
}

template <typename OutT, bool Pow2Scales, int VecSize>
__device__ void BlockStoreScale(float* scale,
                                size_t off_m,
                                __nv_bfloat16 amax[4],
                                float scale_inv[4],
                                size_t K) {
  float scale_out[4];
  for (int i = 0; i < 4; i++) {
    scale_inv[i] = ComputeScale<__nv_bfloat16, OutT, Pow2Scales>(
        static_cast<float>(amax[i]), 0.0f);
    scale_out[i] = __frcp_rn(scale_inv[i]);
  }
  if (threadIdx.y == 0) {
    size_t idx_m = blockIdx.x - off_m / 128;
    size_t off_k = blockIdx.y * 128 + threadIdx.x * VecSize;
    size_t offset = idx_m * K + off_k;

    for (size_t j = 0; j < 4; j += VecSize) {
      if (off_k + j * 32 < K) {
        size_t idx = offset + j * 32;
        using StoreT = VecType<float, VecSize>;
        StoreT data;
        for (int k = 0; k < VecSize; k++) {
          data[k] = scale_out[j + k];
        }
        *reinterpret_cast<StoreT*>(scale + idx) = data;
      }
    }
  }
}

template <typename OutT, int VecSize>
__device__ void BlockStoreOut(OutT* out,
                              size_t off_m,
                              size_t cur_tokens,
                              const OutT shm[128][129],
                              size_t K) {
  for (size_t i = 0; i < 4; i++) {
    size_t idx_m = blockIdx.x * 128 + threadIdx.x * 4;
    size_t idx_k = blockIdx.y * 128 + threadIdx.y + i * 32;
    size_t idx = idx_k * cur_tokens + (idx_m - off_m);

    if (idx_k < K) {
      using StoreT = VecType<OutT, VecSize>;
      StoreT data;
      for (int j = 0; j < VecSize; j++) {
        data[j] = shm[i * 32 + threadIdx.y][threadIdx.x * 4 + j];
      }
      *reinterpret_cast<StoreT*>(out + idx) = data;
    }
  }
}

__device__ std::pair<size_t, size_t> GetExpertIdx(int64_t* tokens_per_expert,
                                                  size_t num_experts) {
  __shared__ size_t expert_idx_, off_m_;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t idx_m = blockIdx.x * 128;
    size_t off_m = 0, next_off_m = 0;
    size_t expert_idx;
    for (expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      next_off_m += tokens_per_expert[expert_idx];
      if (idx_m >= off_m && idx_m < next_off_m) {
        break;
      }
      off_m = next_off_m;
    }
    expert_idx_ = expert_idx;
    off_m_ = off_m;
  }
  __syncthreads();

  return {expert_idx_, off_m_};
}

template <typename OutT, bool Pow2Scales, int VecSize>
__global__ void __launch_bounds__(1024)
    FusedTransposeSplitQuantKernel(const phi::bfloat16* __restrict__ X,
                                   int64_t* __restrict__ meta,
                                   size_t num_experts,
                                   size_t K) {
  __shared__ OutT shm[128][129];
  int64_t* tokens_per_expert = meta;
  OutT** out_ptrs = reinterpret_cast<OutT**>(meta + num_experts);
  float** scale_ptrs = reinterpret_cast<float**>(meta + num_experts * 2);

  // Get expert_idx and offset at the M dim of the current block
  auto expert_info = GetExpertIdx(tokens_per_expert, num_experts);
  size_t expert_idx = expert_info.first;
  size_t off_m = expert_info.second;

  // Load 128x128 elements from X
  __nv_bfloat16 input[4][4];
  BlockLoad<VecSize>(X, input, K);

  // Find the maximum of each 128 elements on the M axis
  __nv_bfloat16 amax[4];
  BlockColumnMax(input, amax, reinterpret_cast<__nv_bfloat16*>(shm));

  // Compute scale and scale_inv, then store scale back
  float scale_inv[4];
  BlockStoreScale<OutT, Pow2Scales, VecSize>(
      scale_ptrs[expert_idx], off_m, amax, scale_inv, K);

  // Scale X and save into shared memory with transposed layout
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j += VecSize) {
      for (int k = 0; k < VecSize; k++) {
        float input_fp32 = static_cast<float>(input[i][j + k]);
        float output_scaled = input_fp32 * scale_inv[j + k];
        shm[threadIdx.x * VecSize + j * 32 + k][i * 32 + threadIdx.y] =
            static_cast<OutT>(output_scaled);
      }
    }
  }
  __syncthreads();

  // Store 128x128 elements back
  // Note: out is always 4x vectorizable.
  BlockStoreOut<OutT, 4>(
      out_ptrs[expert_idx], off_m, tokens_per_expert[expert_idx], shm, K);
}

template <typename T, typename Context>
void FusedTransposeSplitQuantKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const std::vector<int64_t>& tokens_per_expert,
    bool pow_2_scales,
    std::vector<DenseTensor*> outs,
    std::vector<DenseTensor*> scales) {
  auto x_dims = x.dims();
  const int64_t M = x_dims[0];
  const int64_t K = x_dims[1];
  const size_t num_experts = tokens_per_expert.size();

  if (M == 0 || K == 0 || num_experts == 0) {
    return;
  }

  for (size_t i = 0; i < num_experts; i++) {
    if (outs[i] != nullptr) {
      dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(outs[i]);
    }
    if (scales[i] != nullptr) {
      dev_ctx.template Alloc<float>(scales[i]);
    }
  }

  DenseTensor meta_cpu;
  meta_cpu.Resize({static_cast<int64_t>(num_experts * 3)});
  dev_ctx.template HostAlloc<int64_t>(&meta_cpu);

  int64_t* meta_ptr = meta_cpu.data<int64_t>();

  for (size_t i = 0; i < num_experts; i++) {
    meta_ptr[i] = tokens_per_expert[i];
  }

  for (size_t i = 0; i < num_experts; i++) {
    meta_ptr[num_experts + i] =
        outs[i] != nullptr ? reinterpret_cast<int64_t>(
                                 outs[i]->data<phi::dtype::float8_e4m3fn>())
                           : 0;
  }

  for (size_t i = 0; i < num_experts; i++) {
    meta_ptr[num_experts * 2 + i] =
        scales[i] != nullptr
            ? reinterpret_cast<int64_t>(scales[i]->data<float>())
            : 0;
  }

  DenseTensor meta_gpu;
  phi::Copy(dev_ctx, meta_cpu, dev_ctx.GetPlace(), false, &meta_gpu);

  auto stream = dev_ctx.stream();

  dim3 grid(M / 128, (K + 127) / 128);
  dim3 block(32, 32);

#define LAUNCH_KERNEL(POW_2_SCALES, VEC_SIZE)                      \
  FusedTransposeSplitQuantKernel<phi::dtype::float8_e4m3fn,        \
                                 POW_2_SCALES,                     \
                                 VEC_SIZE>                         \
      <<<grid, block, 0, stream>>>(x.data<phi::dtype::bfloat16>(), \
                                   meta_gpu.data<int64_t>(),       \
                                   num_experts,                    \
                                   K);

#define LAUNCH_KERNEL_PARTIAL(VEC_SIZE) \
  if (pow_2_scales) {                   \
    LAUNCH_KERNEL(true, VEC_SIZE);      \
  } else {                              \
    LAUNCH_KERNEL(false, VEC_SIZE);     \
  }

  if (K % 4 == 0) {
    LAUNCH_KERNEL_PARTIAL(4);
  } else if (K % 2 == 0) {
    LAUNCH_KERNEL_PARTIAL(2);
  } else {
    LAUNCH_KERNEL_PARTIAL(1);
  }

#undef LAUNCH_KERNEL_PARTIAL
#undef LAUNCH_KERNEL

  PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_transpose_split_quant,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedTransposeSplitQuantKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT8_E4M3FN);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
}
