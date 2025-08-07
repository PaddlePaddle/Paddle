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

template <typename InT, int VecSize>
__device__ void BlockLoad(const InT* input,
                          const float* input_scales,
                          __nv_bfloat16 x[8][4],
                          size_t K) {
  constexpr bool need_dequant = std::is_same_v<InT, phi::dtype::float8_e4m3fn>;
  for (uint32_t i = 0; i < 8; i++) {
    size_t local_off_M = threadIdx.y + i * 16;
    size_t off_m = blockIdx.x * size_t(128) + local_off_M;
    size_t off_k = blockIdx.y * 128 + threadIdx.x * VecSize;
    size_t offset = off_m * K + off_k;
    float scale;
    if constexpr (need_dequant) {
      scale = input_scales[local_off_M];
    }

    for (uint32_t j = 0; j < 4; j += VecSize) {
      if (off_k + j * 32 < K) {
        size_t idx = offset + j * 32;
        using LoadT = VecType<InT, VecSize>;
        LoadT data = *reinterpret_cast<const LoadT*>(input + idx);
        for (uint32_t k = 0; k < VecSize; k++) {
          if constexpr (need_dequant) {
            x[i][j + k] = __float2bfloat16(static_cast<float>(data[k]) * scale);
          } else {
            x[i][j + k] = (*reinterpret_cast<__nv_bfloat16*>(&data[k]));
          }
        }
      }
    }
  }
}

template <bool Pow2Scales>
__device__ void BlockColumnScale(const __nv_bfloat16 x[8][4],
                                 float scales[128],
                                 __nv_bfloat16* shm) {
  // reduce [(8), 16, 32, 4] => [16, 32, 4]
  __nv_bfloat16 warp_max[4];
  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 4; j++) {
      __nv_bfloat16 t = BF16_ABS(x[i][j]);
      warp_max[j] = i == 0 ? t : BF16_MAX(warp_max[j], t);
    }
  }

  // reduce [(16), 32, 4] => [8, 32, 4]
  if (threadIdx.y >= 8) {
    for (uint32_t j = 0; j < 4; j++) {
      shm[(threadIdx.y - 8) * 128 + threadIdx.x + j * 32] = warp_max[j];
    }
  }
  __syncthreads();

  // reduce [(8), 32, 4] => [32, 4]
  for (uint32_t offset = 8; offset > 0; offset /= 2) {
    if (threadIdx.y < offset) {
      for (uint32_t j = 0; j < 4; j++) {
        __nv_bfloat16 other =
            offset == 8
                ? warp_max[j]
                : shm[(threadIdx.y + offset) * 128 + threadIdx.x + j * 32];
        __nv_bfloat16 next_val =
            BF16_MAX(shm[threadIdx.y * 128 + threadIdx.x + j * 32], other);
        if (offset > 1) {
          shm[threadIdx.y * 128 + threadIdx.x + j * 32] = next_val;
        } else {
          scales[threadIdx.x + j * 32] =
              ComputeScale<__nv_bfloat16, __nv_fp8_e4m3, Pow2Scales>(
                  static_cast<float>(next_val), 0.0f);
        }
      }
    }
    __syncthreads();
  }
}

template <typename OutT, int VecSize>
__device__ void BlockStoreScale(float* scale,
                                size_t off_m,
                                float scales[128],
                                size_t K) {
  if (threadIdx.y < 4) {
    uint32_t off = threadIdx.y * 32 + threadIdx.x;
    if constexpr (VecSize == 4) {
      off = (off % 4) * 32 + off / 4;
    } else if constexpr (VecSize == 2) {
      off = (off / 64) * 64 + (off % 2) * 32 + (off % 64) / 2;
    }
    float scale_out = 1.0f / scales[off];
    size_t idx_y = blockIdx.x - off_m / 128;
    size_t idx_x = blockIdx.y * 128 + threadIdx.y * 32 + threadIdx.x;
    size_t idx = idx_y * K + idx_x;
    if (idx_x < K) {
      scale[idx] = scale_out;
    }
  }
}

template <typename OutT, int VecSize>
__device__ void BlockStoreOut(OutT* out,
                              size_t off_m,
                              size_t cur_tokens,
                              const OutT shm[128][129],
                              size_t K) {
  for (uint32_t i = 0; i < 8; i++) {
    size_t idx_m = blockIdx.x * size_t(128) + threadIdx.x * 4;
    size_t idx_k = blockIdx.y * 128 + threadIdx.y + i * 16;
    size_t idx = idx_k * cur_tokens + (idx_m - off_m);

    if (idx_k < K) {
      using StoreT = VecType<OutT, VecSize>;
      StoreT data;
      for (uint32_t j = 0; j < VecSize; j++) {
        data[j] = shm[i * 16 + threadIdx.y][threadIdx.x * 4 + j];
      }
      *reinterpret_cast<StoreT*>(out + idx) = data;
    }
  }
}

template <typename InT, typename OutT, bool Pow2Scales, int VecSize>
__global__ void __launch_bounds__(512)
    FusedTransposeSplitQuantKernel(const InT* __restrict__ input,
                                   const float* __restrict__ input_scales,
                                   int64_t* __restrict__ meta,
                                   size_t num_experts,
                                   size_t K,
                                   size_t k_scaled) {
  __shared__ OutT shm[128][129];
  __shared__ float scales[128];
  __shared__ size_t expert_info[2];

  // 0. Load input_scales if float8
  if constexpr (std::is_same_v<InT, phi::dtype::float8_e4m3fn>) {
    const int tid = blockDim.y * threadIdx.x + threadIdx.y;
    if (tid < 128) {
      const size_t m_base = blockIdx.x * size_t(128);
      const size_t k_base = blockIdx.y * size_t(128);
      const size_t m_stride = k_scaled;
      const size_t input_scale_offset = (m_base + tid) * m_stride + blockIdx.y;
      scales[tid] = input_scales[input_scale_offset];
    }
    __syncthreads();
  }
  int64_t* tokens_per_expert = meta;
  OutT** out_ptrs = reinterpret_cast<OutT**>(meta + num_experts);
  float** scale_ptrs = reinterpret_cast<float**>(meta + num_experts * 2);

  // 1. Load 128x128 elements from input
  __nv_bfloat16 x[8][4];
  BlockLoad<InT, VecSize>(input, scales, x, K);

  // 2. Get expert index and offset of the current block
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t idx_m = blockIdx.x * size_t(128);
    size_t off_m = 0, next_off_m = 0;
    size_t expert_idx;
    for (expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      next_off_m += tokens_per_expert[expert_idx];
      if (idx_m >= off_m && idx_m < next_off_m) {
        break;
      }
      off_m = next_off_m;
    }
    expert_info[0] = expert_idx;
    expert_info[1] = off_m;
  }

  if constexpr (std::is_same_v<InT, phi::dtype::float8_e4m3fn>) {
    __syncthreads();
  }

  // 3. Calculate scale along the column
  BlockColumnScale<Pow2Scales>(
      x, scales, reinterpret_cast<__nv_bfloat16*>(shm));

  // 4. Store scale
  const size_t expert_idx = expert_info[0];
  const size_t off_m = expert_info[1];
  BlockStoreScale<OutT, VecSize>(scale_ptrs[expert_idx], off_m, scales, K);

  // 5. Scale x and save into shared memory with transposed layout
  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 4; j += VecSize) {
      for (uint32_t k = 0; k < VecSize; k++) {
        float x_fp32 = static_cast<float>(x[i][j + k]);
        float x_scaled = x_fp32 * scales[threadIdx.x + (j + k) * 32];
        shm[threadIdx.x * VecSize + j * 32 + k][i * 16 + threadIdx.y] =
            static_cast<OutT>(x_scaled);
      }
    }
  }
  __syncthreads();

  // 6. Store 128x128 elements back
  // Note: out is always 4x vectorizable.
  BlockStoreOut<OutT, 4>(
      out_ptrs[expert_idx], off_m, tokens_per_expert[expert_idx], shm, K);
}

template <typename T, typename Context>
void FusedTransposeSplitQuantKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& input_scales,
    const std::vector<int64_t>& tokens_per_expert,
    bool pow_2_scales,
    std::vector<DenseTensor*> outs,
    std::vector<DenseTensor*> output_scales) {
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
    if (output_scales[i] != nullptr) {
      dev_ctx.template Alloc<float>(output_scales[i]);
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
        output_scales[i] != nullptr
            ? reinterpret_cast<int64_t>(output_scales[i]->data<float>())
            : 0;
  }

  DenseTensor meta_gpu;
  phi::Copy(dev_ctx, meta_cpu, dev_ctx.GetPlace(), false, &meta_gpu);

  auto stream = dev_ctx.stream();

  // pre-compute on CPU to reduce size_t division cost in kernel
  const size_t k_scaled = (K + 127) / 128;
  dim3 grid(M / 128, k_scaled);
  dim3 block(32, 16);

#define DTYPE_CASE(dtype, type) dtype == phi::DataType::type
#define LAUNCH_KERNEL(T, POW_2_SCALES, VEC_SIZE)                        \
  FusedTransposeSplitQuantKernel<T,                                     \
                                 phi::dtype::float8_e4m3fn,             \
                                 POW_2_SCALES,                          \
                                 VEC_SIZE><<<grid, block, 0, stream>>>( \
      x.data<T>(),                                                      \
      input_scales ? input_scales.get_ptr()->data<float>() : nullptr,   \
      meta_gpu.data<int64_t>(),                                         \
      num_experts,                                                      \
      K,                                                                \
      k_scaled);
#define DISPATCH_DATATYPE(POW_2_SCALES, VEC_SIZE)              \
  if (DTYPE_CASE(x.dtype(), BFLOAT16)) {                       \
    LAUNCH_KERNEL(phi::bfloat16, POW_2_SCALES, VEC_SIZE);      \
  } else if (DTYPE_CASE(x.dtype(), FLOAT8_E4M3FN)) {           \
    LAUNCH_KERNEL(phi::float8_e4m3fn, POW_2_SCALES, VEC_SIZE); \
  }

#define LAUNCH_KERNEL_PARTIAL(VEC_SIZE) \
  if (pow_2_scales) {                   \
    DISPATCH_DATATYPE(true, VEC_SIZE);  \
  } else {                              \
    DISPATCH_DATATYPE(false, VEC_SIZE); \
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
                   phi::dtype::bfloat16,
                   phi::dtype::float8_e4m3fn) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT8_E4M3FN);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
}
