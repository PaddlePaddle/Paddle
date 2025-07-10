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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"
#include "paddle/phi/kernels/legacy/gpu/moe_fuse_op.h"
#include "paddle/phi/kernels/legacy/gpu/moe_ops_utils.h"

namespace phi {
constexpr int64_t TileSize = 128;
constexpr int64_t WarpSize = 32;

#define LAUNCH_KERNEL(ELEMENTS_PER_THREAD, THREADS, POWER2SCALING)           \
  initialize_moe_routing_kernel<ELEMENTS_PER_THREAD, THREADS, POWER2SCALING> \
      <<<blocks, THREADS, 0, stream>>>(                                      \
          unpermuted_input,                                                  \
          permuted_output,                                                   \
          scale_out,                                                         \
          expanded_dest_row_to_expanded_source_row,                          \
          expanded_source_row_to_expanded_dest_row,                          \
          permuted_experts,                                                  \
          expert_offset,                                                     \
          combine_weights,                                                   \
          num_rows,                                                          \
          cols,                                                              \
          k,                                                                 \
          capacity,                                                          \
          num_experts,                                                       \
          use_pad)

template <bool Power2Scaling>
__device__ __forceinline__ float ScaleWrapper(const float amax,
                                              const float eps = 0.f) {
  constexpr float fp8_max = 448.0f;
  float amax_mod = fmaxf(amax, eps);
  if (amax_mod == 0.f) {
    return 1.f;
  }
  float scale = fp8_max / amax_mod;

  if (isinf(scale)) {
    if constexpr (Power2Scaling) {
      return 0x1.0p127;
    } else {
      return 0x1.FEp127;
    }
  }

  if (scale == 0.0) {
    return scale;
  }
  if constexpr (Power2Scaling) {
    uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&scale);
    uint8_t exp = scale_bits >> 23;
    int32_t normal_biased_exp = static_cast<int32_t>(exp) - 127;
    __builtin_assume(exp != 0);
    scale = ldexpf(1.0f, normal_biased_exp);
  }
  return scale;
}

template <int VecSize, bool Power2Scaling>
__device__ void ComputeScaleAndWrite(__nv_bfloat16 *data,
                                     float *scale,
                                     float *scale_out,
                                     int64_t local_scale_id,
                                     int64_t dest_scale_row,
                                     int64_t dest_scale_col,
                                     int64_t scale_row_num,
                                     int64_t scale_col_num) {
  // -------------------------------------------------------------------------
  // Step 1: Compute local maximum within each thread's vector
  // -------------------------------------------------------------------------
  __nv_bfloat16 local_max = __float2bfloat16(-INFINITY);
  for (int i = 0; i < VecSize; ++i) {
    __nv_bfloat16 val = BF16_ABS(data[i]);
    local_max = BF16_MAX(val, local_max);
  }

  // -------------------------------------------------------------------------
  // Step 2: Reduce maximum across warp using shuffle operations
  // -------------------------------------------------------------------------
  static_assert(VecSize >= 4,
                "VecSize must be at least 4 to avoid cross-warp reduction");
  static_assert(TileSize >= VecSize && TileSize % VecSize == 0,
                "TileSize must be >= VecSize and a multiple of VecSize");

  __nv_bfloat16 global_max = local_max;
  constexpr int group_size = TileSize / VecSize;  // Elements per thread group
  const int lane_id = threadIdx.x % WarpSize;
  const int group_id = lane_id / group_size;
  const int group_lane = lane_id % group_size;
  const unsigned mask =
      (1u << ((group_id + 1) * group_size)) - (1u << (group_id * group_size));

  // Parallel reduction within each group
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    __nv_bfloat16 other = __shfl_down_sync(mask, global_max, stride);
    global_max = BF16_MAX(other, global_max);
  }

  // -------------------------------------------------------------------------
  // Step 3: Write the computed scale to shared and global memory
  // -------------------------------------------------------------------------
  if (group_lane == 0) {
    float scale_value =
        ScaleWrapper<Power2Scaling>(__bfloat162float(global_max));

    // Write to shared memory
    scale[local_scale_id] = scale_value;

    // Write inverse to global memory (CUTLASS layout - no transpose needed)
    scale_out[dest_scale_row * scale_col_num + dest_scale_col] =
        1.0f / scale_value;
  }

  __syncthreads();
}

template <int VecSize>
__device__ void ApplyScale(const __nv_bfloat16 *src_ptr,
                           __nv_fp8_e4m3 *dest_ptr,
                           const float *scale,
                           int64_t scale_id) {
  const __nv_bfloat16 *data = reinterpret_cast<const __nv_bfloat16 *>(src_ptr);
  float scale_value = scale[scale_id];
  for (int64_t i = 0; i < VecSize; i++) {
    dest_ptr[i] =
        static_cast<__nv_fp8_e4m3>(static_cast<float>(data[i]) * scale_value);
  }
}

template <int VecSize, int ThreadNum, bool Power2Scaling>
__global__ void initialize_moe_routing_kernel(
    const __nv_bfloat16 *unpermuted_input,
    __nv_fp8_e4m3 *permuted_output,
    float *scale_out,
    const int *expanded_dest_row_to_expanded_source_row,
    int *expanded_source_row_to_expanded_dest_row,
    const int *permuted_experts,
    const int64_t *expert_offset,
    float *combine_weights,  // output
    const int num_rows,
    const int cols,
    const int k,
    const int64_t capacity,
    const int64_t num_experts,
    bool use_pad) {
  // Static assertions for compile-time checks
  static_assert(VecSize * ThreadNum % TileSize == 0,
                "VecSize * ThreadNum must be a multiple of TileSize");
  static_assert(VecSize <= TileSize,
                "VecSize must be less than or equal to TileSize");

  using LoadT = phi::AlignedVector<__nv_bfloat16, VecSize>;
  using StoreT = phi::AlignedVector<__nv_fp8_e4m3, VecSize>;
  LoadT src_vec;
  StoreT dest_vec;

  const int64_t expanded_dest_row = blockIdx.x;
  const int64_t expanded_source_row =
      expanded_dest_row_to_expanded_source_row[expanded_dest_row];
  const int64_t iexpert = permuted_experts[expanded_dest_row];
  const int64_t offset = iexpert == 0 ? 0 : (expert_offset[iexpert - 1]);
  const int64_t row_in_expert = expanded_dest_row - offset;
  if (row_in_expert >= capacity) {
    if (threadIdx.x == 0) {
      expanded_source_row_to_expanded_dest_row[expanded_source_row] =
          0;  // unset scatter-idx
      auto ik = expanded_source_row / num_rows;
      auto isent = expanded_source_row % num_rows;  // transpose
      combine_weights[isent * k + ik] = 0.f;        // unset combine-weight
    }
    return;
  }
  int64_t num_padded = 0;
  if (threadIdx.x == 0) {
    if (use_pad) num_padded = iexpert * capacity - offset;
    expanded_source_row_to_expanded_dest_row[expanded_source_row] =
        expanded_dest_row + num_padded;
  }
  // Duplicate and permute rows
  const int64_t source_row = expanded_source_row % num_rows;

  const __nv_bfloat16 *source_row_ptr = unpermuted_input + source_row * cols;
  __nv_fp8_e4m3 *dest_row_ptr;
  float *scale_out_prt;
  int64_t dest_row;
  int64_t dest_row_num;
  if (use_pad) {
    dest_row = iexpert * capacity + row_in_expert;
    dest_row_num = num_experts * capacity;
  } else {
    dest_row = expanded_dest_row;
    dest_row_num = num_rows * k;
  }
  dest_row_ptr = permuted_output + dest_row * cols;

  __shared__ float scale[ThreadNum * VecSize / TileSize];

  for (int64_t element_id = threadIdx.x * VecSize; element_id < cols;
       element_id += blockDim.x * VecSize) {
    // Each thread reads VecSize elements, totaling ThreadNum*VecSize elements
    // read Note: A single thread can compute at most one scale value
    phi::Load<__nv_bfloat16, VecSize>(&source_row_ptr[element_id], &src_vec);

    int64_t local_scale_id = VecSize * threadIdx.x / TileSize;

    ComputeScaleAndWrite<VecSize, Power2Scaling>(src_vec.val,
                                                 scale,
                                                 scale_out,
                                                 local_scale_id,
                                                 dest_row,
                                                 element_id / TileSize,
                                                 dest_row_num,
                                                 cols / TileSize);
    ApplyScale<VecSize>(src_vec.val, dest_vec.val, scale, local_scale_id);

    phi::Store<__nv_fp8_e4m3, VecSize>(dest_vec, &dest_row_ptr[element_id]);
  }
}

void fp8_initialize_moe_routing_kernelLauncher(
    const __nv_bfloat16 *unpermuted_input,
    __nv_fp8_e4m3 *permuted_output,
    float *scale_out,
    const int *expanded_dest_row_to_expanded_source_row,
    int *expanded_source_row_to_expanded_dest_row,
    const int *permuted_experts,
    const int64_t *expert_offset,
    float *combine_weights,  // output
    const int num_rows,
    const int cols,
    const int k,
    const int64_t capacity,
    const int64_t num_experts,
    bool use_pad,
    bool use_pow2_scale,
    cudaStream_t stream) {
  const int blocks = num_rows * k;
  DISPATCH_BOOL(
      use_pow2_scale,
      k_use_pow2_scale,
      if (cols % 2048 == 0) {
        constexpr int threads = 256;
        LAUNCH_KERNEL(8, threads, k_use_pow2_scale);
      } else if (cols % 1024 == 0) {
        constexpr int threads = 256;
        LAUNCH_KERNEL(4, threads, k_use_pow2_scale);
      } else if (cols % 512 == 0) {
        constexpr int threads = 128;
        LAUNCH_KERNEL(4, threads, k_use_pow2_scale);
      } else if (cols % 256 == 0) {
        constexpr int threads = 64;
        LAUNCH_KERNEL(4, threads, k_use_pow2_scale);
      } else if (cols % 128 == 0) {
        constexpr int threads = 32;
        LAUNCH_KERNEL(4, threads, k_use_pow2_scale);
      } else { assert(0); })
}

template <typename T, typename Context>
void apply_moe_dispatch_fwd(const Context &dev_ctx,
                            const T *x,
                            const float *gate_logits,
                            const float *corr_bias,
                            int64_t num_rows,
                            int64_t num_experts,
                            int64_t hidden_size,
                            int64_t capacity,
                            int64_t k,
                            __nv_fp8_e4m3 *out_fp8,
                            float *out_scale,
                            float *combine_weights,
                            int *scatter_index,
                            int64_t *expert_offset,
                            int *expert_id,
                            bool use_pad,
                            bool use_pow2_scale,
                            cudaStream_t stream) {
  int *permuted_rows = nullptr;
  int *permuted_experts = nullptr;
  topk_gating(dev_ctx,
              x,
              gate_logits,
              corr_bias,
              &permuted_rows,
              &permuted_experts,
              num_rows,
              num_experts,
              hidden_size,
              capacity,
              k,
              combine_weights,
              scatter_index,
              expert_offset,
              expert_id,
              use_pad,
              stream);

  fp8_initialize_moe_routing_kernelLauncher(
      reinterpret_cast<const __nv_bfloat16 *>(x),
      out_fp8,
      out_scale,
      permuted_rows,
      scatter_index,
      permuted_experts,
      expert_offset,
      combine_weights,
      static_cast<int>(num_rows),
      static_cast<int>(hidden_size),
      static_cast<int>(k),
      capacity,
      num_experts,
      use_pad,
      use_pow2_scale,
      stream);

  return;
}

template <typename T, typename Context>
void MoeDispatchAndQuantKernel(const Context &dev_ctx,
                               const DenseTensor &x,
                               const DenseTensor &gate_logits,
                               const paddle::optional<DenseTensor> &corr_bias,
                               int64_t k,
                               int64_t capacity,
                               bool use_pad,
                               bool use_pow2_scale,
                               DenseTensor *out_fp8,
                               DenseTensor *scale,
                               DenseTensor *combine_weights,
                               DenseTensor *scatter_index,
                               DenseTensor *expert_offset,
                               DenseTensor *expert_id) {
  dev_ctx.template Alloc<int>(expert_id);
  dev_ctx.template Alloc<int64_t>(expert_offset);
  dev_ctx.template Alloc<int>(scatter_index);
  dev_ctx.template Alloc<float>(combine_weights);
  dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(out_fp8);
  dev_ctx.template Alloc<float>(scale);

  cudaMemsetAsync(
      reinterpret_cast<void *>(out_fp8->data<phi::dtype::float8_e4m3fn>()),
      0,
      sizeof(phi::dtype::float8_e4m3fn) * out_fp8->numel(),
      dev_ctx.stream());

  phi::Full<float, Context>(
      dev_ctx, phi::IntArray(common::vectorize(scale->dims())), 1, scale);

  const auto &x_shape = x.dims();
  const auto &gate_logits_shape = gate_logits.dims();
  const int64_t num_rows = x_shape[0];
  const int64_t hidden_size = x_shape[1];
  const int64_t num_experts = gate_logits_shape[1];

  apply_moe_dispatch_fwd<T, Context>(
      dev_ctx,
      x.data<T>(),
      gate_logits.data<float>(),
      corr_bias ? corr_bias.get().data<float>() : nullptr,
      num_rows,
      num_experts,
      hidden_size,
      capacity,
      k,
      reinterpret_cast<__nv_fp8_e4m3 *>(
          out_fp8->data<phi::dtype::float8_e4m3fn>()),
      scale->data<float>(),
      combine_weights->data<float>(),
      scatter_index->data<int>(),
      expert_offset->data<int64_t>(),
      expert_id->data<int>(),
      use_pad,
      use_pow2_scale,
      dev_ctx.stream());
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_gate_dispatch_and_quant,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeDispatchAndQuantKernel,
                   phi::dtype::bfloat16) {}
