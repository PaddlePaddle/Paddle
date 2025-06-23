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
#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"
#include "paddle/phi/kernels/primitive/datamover_primitives.h"

namespace phi {
namespace fusion {

using FastDivMod = phi::funcs::FastDivMod<int64_t>;

template <bool Pow2Scales>
__device__ void BlockColumnScale(const __nv_bfloat16 x[8][4],
                                 float col_scale[128],
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
          col_scale[threadIdx.x + j * 32] =
              ComputeScale<__nv_bfloat16, __nv_fp8_e4m3, Pow2Scales>(
                  static_cast<float>(next_val), 0.0f);
        }
      }
    }
    __syncthreads();
  }
}

// Permute [L, W, C] => [W, L, C]
__device__ size_t PermuteHighIdx(const size_t idx,
                                 const FastDivMod W,
                                 const size_t L,
                                 const FastDivMod C) {
  auto split_idx = C.Divmod(idx);
  size_t idx_lw = split_idx[0];
  size_t idx_c = split_idx[1];
  auto split_lw = W.Divmod(idx_lw);
  size_t idx_l = split_lw[0];
  size_t idx_w = split_lw[1];
  return (idx_w * L + idx_l) * C.divisor + idx_c;
}

template <bool Pow2Scales, bool BlockwiseC, int VecSize>
__global__ void __launch_bounds__(512)
    FusedTransposeWLCHSplitQuantKernel(const __nv_bfloat16* __restrict__ input,
                                       int64_t* __restrict__ meta,
                                       const size_t num_experts,
                                       const FastDivMod W_divmod,
                                       const size_t L,
                                       const FastDivMod C_divmod,
                                       const size_t H) {
  __nv_bfloat16 x[8][4];
  __shared__ __nv_fp8_e4m3 shm[128][129];
  __shared__ float col_scale[128];
  __shared__ size_t expert_info[2];

  int64_t* tokens_per_expert = meta;
  __nv_fp8_e4m3** out_ptrs =
      reinterpret_cast<__nv_fp8_e4m3**>(meta + num_experts);
  float** scale_ptrs = reinterpret_cast<float**>(meta + num_experts * 2);

  const size_t block_off_x = blockIdx.x * size_t(128);
  const size_t block_off_y = blockIdx.y * 128;

  // 1. Load 128x128 block from input.
  for (uint32_t i = 0; i < 8; i++) {
    size_t idx_y;
    if constexpr (BlockwiseC) {
      idx_y = PermuteHighIdx(block_off_x, W_divmod, L, C_divmod) + threadIdx.y +
              i * 16;
    } else {
      idx_y = PermuteHighIdx(
          block_off_x + threadIdx.y + i * 16, W_divmod, L, C_divmod);
    }
    size_t idx_x = block_off_y + threadIdx.x * VecSize;
    size_t idx = idx_y * H + idx_x;
    for (uint32_t j = 0; j < 4; j += VecSize) {
      if (idx_x + j * 32 < H) {
        using LoadT = phi::kps::details::VectorType<__nv_bfloat16, VecSize>;
        LoadT data = *reinterpret_cast<const LoadT*>(input + idx + j * 32);
        for (uint32_t k = 0; k < VecSize; k++) {
          x[i][j + k] = data.val[k];
        }
      }
    }
  }

  // 2. Get expert index and offset of the current block.
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t expert_off = 0, next_expert_off = 0;
    size_t expert_idx;
    for (expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      next_expert_off += tokens_per_expert[expert_idx];
      if (block_off_x >= expert_off && block_off_x < next_expert_off) {
        break;
      }
      expert_off = next_expert_off;
    }
    expert_info[0] = expert_idx;
    expert_info[1] = expert_off;
  }

  // 3. Calculate scale along the column.
  BlockColumnScale<Pow2Scales>(
      x, col_scale, reinterpret_cast<__nv_bfloat16*>(shm));

  // 4. Store scale.
  const size_t expert_idx = expert_info[0];
  const size_t expert_off = expert_info[1];
  if (threadIdx.y < 4) {
    uint32_t off = threadIdx.y * 32 + threadIdx.x;
    if constexpr (VecSize == 4) {
      off = (off % 4) * 32 + off / 4;
    } else if constexpr (VecSize == 2) {
      off = (off / 64) * 64 + (off % 2) * 32 + (off % 64) / 2;
    }
    float scale_out = 1.0f / col_scale[off];
    size_t idx_y = blockIdx.x - expert_off / 128;
    size_t idx_x = block_off_y + threadIdx.y * 32 + threadIdx.x;
    size_t idx = idx_y * H + idx_x;
    if (idx_x < H) {
      scale_ptrs[expert_idx][idx] = scale_out;
    }
  }

  // 5. Quantize x and transpose using shared memory.
  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 4; j += VecSize) {
      for (uint32_t k = 0; k < VecSize; k++) {
        float x_cast = static_cast<float>(x[i][j + k]);
        float x_scaled = x_cast * col_scale[threadIdx.x + (j + k) * 32];
        shm[threadIdx.x * VecSize + j * 32 + k][threadIdx.y + i * 16] =
            static_cast<__nv_fp8_e4m3>(x_scaled);
      }
    }
  }
  __syncthreads();

  // 6. Store 128x128 output.
  const size_t cur_tokens = tokens_per_expert[expert_idx];
  __nv_fp8_e4m3* out = out_ptrs[expert_idx];
  for (uint32_t i = 0; i < 8; i++) {
    size_t idx_y = block_off_y + threadIdx.y + i * 16;
    size_t idx_x = block_off_x + threadIdx.x * 4;
    size_t idx = idx_y * cur_tokens + (idx_x - expert_off);
    if (idx_y < H) {
      // Note: out is always 4x vectorizable.
      using StoreT = phi::kps::details::VectorType<__nv_fp8_e4m3, 4>;
      StoreT data;
      for (uint32_t j = 0; j < 4; j++) {
        data.val[j] = shm[threadIdx.y + i * 16][threadIdx.x * 4 + j];
      }
      *reinterpret_cast<StoreT*>(out + idx) = data;
    }
  }
}

template <typename T, typename Context>
void FusedTransposeWLCHSplitQuantKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const std::vector<int64_t>& tokens_per_expert,
    bool pow_2_scales,
    std::vector<DenseTensor*> outs,
    std::vector<DenseTensor*> scales) {
  const auto x_dims = x.dims();
  const int64_t W = x_dims[0];
  const int64_t L = x_dims[1];
  const int64_t C = x_dims[2];
  const int64_t H = x_dims[3];
  const size_t num_experts = tokens_per_expert.size();

  // Allocate outs and scales
  for (size_t i = 0; i < num_experts; i++) {
    if (outs[i] != nullptr) {
      dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(outs[i]);
    }
    if (scales[i] != nullptr) {
      dev_ctx.template Alloc<float>(scales[i]);
    }
  }

  if (W == 0 || L == 0 || C == 0 || H == 0) {
    return;
  }

  // Copy tokens_per_expert and pointers of outs & scales to device
  DenseTensor meta_cpu;
  meta_cpu.Resize({static_cast<int64_t>(num_experts * 3)});
  dev_ctx.template HostAlloc<int64_t>(&meta_cpu);

  int64_t* meta_ptr = meta_cpu.data<int64_t>();
  for (size_t i = 0; i < num_experts; i++) {
    meta_ptr[i] = tokens_per_expert[i];
  }
  for (size_t i = 0; i < num_experts; i++) {
    meta_ptr[num_experts + i] =
        outs[i] ? reinterpret_cast<int64_t>(
                      outs[i]->data<phi::dtype::float8_e4m3fn>())
                : 0;
  }
  for (size_t i = 0; i < num_experts; i++) {
    meta_ptr[num_experts * 2 + i] =
        scales[i] ? reinterpret_cast<int64_t>(scales[i]->data<float>()) : 0;
  }

  DenseTensor meta_gpu;
  phi::Copy(dev_ctx, meta_cpu, dev_ctx.GetPlace(), false, &meta_gpu);

  // Launch kernel
  auto stream = dev_ctx.stream();
  dim3 grid(W * L * C / 128, (H + 127) / 128);
  dim3 block(32, 16);

  const __nv_bfloat16* x_ptr =
      reinterpret_cast<const __nv_bfloat16*>(x.data<phi::dtype::bfloat16>());
  int64_t* meta_gpu_ptr = meta_gpu.data<int64_t>();
  FastDivMod W_divmod(W), C_divmod(C);

#define LAUNCH_KERNEL(VecSize)                                        \
  FusedTransposeWLCHSplitQuantKernel<Pow2Scales, BlockwiseC, VecSize> \
      <<<grid, block, 0, stream>>>(                                   \
          x_ptr, meta_gpu_ptr, num_experts, W_divmod, L, C_divmod, H)

  DISPATCH_BOOL(
      pow_2_scales, Pow2Scales, DISPATCH_BOOL(C % 128 == 0, BlockwiseC, {
        if (H % 4 == 0) {
          LAUNCH_KERNEL(4);
        } else if (H % 2 == 0) {
          LAUNCH_KERNEL(2);
        } else {
          LAUNCH_KERNEL(1);
        }
      }));

#undef LAUNCH_KERNEL
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_transpose_wlch_split_quant,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedTransposeWLCHSplitQuantKernel,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT8_E4M3FN);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
}
