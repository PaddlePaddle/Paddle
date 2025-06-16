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

#include <cuda_runtime.h>
#include <vector>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
using BFloat16 = __nv_bfloat16;
#else
struct BFloat16 {
  uint16_t x;

  __host__ __device__ BFloat16() : x(0) {}

  __host__ __device__ BFloat16(float val) {
    uint32_t* val_bits = reinterpret_cast<uint32_t*>(&val);
    x = static_cast<uint16_t>(*val_bits >> 16);
  }

  __host__ __device__ operator float() const {
    uint32_t bits = static_cast<uint32_t>(x) << 16;
    return *reinterpret_cast<float*>(&bits);
  }
};
#endif

template <int thread_per_block>
__global__ void SwigluProbsGradKernel(
    const BFloat16* o1,           // [seq_len*topk, moe_intermediate_size*2]
    const BFloat16* do2_s,        // [seq_len*topk, moe_intermediate_size]
    const float* unzipped_probs,  // [seq_len*topk, 1]
    BFloat16* do1,                // [seq_len*topk, moe_intermediate_size*2]
    float* probs_grad,            // [seq_len*topk, 1]
    BFloat16* o2_s,               // [seq_len*topk, moe_intermediate_size]
    int moe_intermediate_size) {
  const int row_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const BFloat16* o1_row = o1 + row_idx * moe_intermediate_size * 2;
  const BFloat16* do2_s_row = do2_s + row_idx * moe_intermediate_size;
  BFloat16* do1_row = do1 + row_idx * moe_intermediate_size * 2;
  BFloat16* o2s_row = o2_s + row_idx * moe_intermediate_size;

  float prob = unzipped_probs[row_idx];

  __shared__ float sum_buffer[thread_per_block];

  float local_probs_grad = 0.0f;

  for (int i = tid; i < moe_intermediate_size; i += blockDim.x) {
    float lhs = static_cast<float>(o1_row[i]);
    float rhs = static_cast<float>(o1_row[i + moe_intermediate_size]);

    float sig = 1.0f / (1.0f + expf(-lhs));
    float tmp = sig * lhs;
    float o2_val = tmp * rhs;

    float do2_s_val = static_cast<float>(do2_s_row[i]);
    float do2_val = do2_s_val * prob;

    float x0_grad = do2_val * rhs * sig * (1.0f + lhs - tmp);
    float x1_grad = do2_val * tmp;

    do1_row[i] = BFloat16(x0_grad);
    do1_row[i + moe_intermediate_size] = BFloat16(x1_grad);
    o2s_row[i] = BFloat16(o2_val * prob);

    local_probs_grad += do2_s_val * o2_val;
  }

  sum_buffer[tid] = local_probs_grad;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_buffer[tid] += sum_buffer[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    probs_grad[row_idx] = sum_buffer[0];
  }
}

typedef struct __align__(8) {
  __nv_bfloat16 x;
  __nv_bfloat16 y;
  __nv_bfloat16 z;
  __nv_bfloat16 w;
}
bfloat16x4_t;

__device__ __forceinline__ float4 fast_swiglu_vec4(const bfloat16x4_t& lhs,
                                                   const bfloat16x4_t& rhs) {
  const float x_f_x = __bfloat162float(lhs.x);
  const float x_f_y = __bfloat162float(lhs.y);
  const float x_f_z = __bfloat162float(lhs.z);
  const float x_f_w = __bfloat162float(lhs.w);

  const float y_f_x = __bfloat162float(rhs.x);
  const float y_f_y = __bfloat162float(rhs.y);
  const float y_f_z = __bfloat162float(rhs.z);
  const float y_f_w = __bfloat162float(rhs.w);

  const float silu_x = x_f_x * __frcp_rn(1.0f + __expf(-x_f_x));
  const float silu_y = x_f_y * __frcp_rn(1.0f + __expf(-x_f_y));
  const float silu_z = x_f_z * __frcp_rn(1.0f + __expf(-x_f_z));
  const float silu_w = x_f_w * __frcp_rn(1.0f + __expf(-x_f_w));

  return {silu_x * y_f_x, silu_y * y_f_y, silu_z * y_f_z, silu_w * y_f_w};
}

__device__ __forceinline__ float4 f4_prod(const float4& x_f,
                                          const float4& y_f) {
  return {x_f.x * y_f.x, x_f.y * y_f.y, x_f.z * y_f.z, x_f.w * y_f.w};
}
__device__ __forceinline__ float4 f4_prod(const float4& x_f, const float& y_f) {
  return {x_f.x * y_f, x_f.y * y_f, x_f.z * y_f, x_f.w * y_f};
}
__device__ __forceinline__ float4 f4_add(const float4& x_f, const float& y_f) {
  return {x_f.x + y_f, x_f.y + y_f, x_f.z + y_f, x_f.w + y_f};
}
__device__ __forceinline__ float4 f4_add(const float4& x_f, const float4& y_f) {
  return {x_f.x + y_f.x, x_f.y + y_f.y, x_f.z + y_f.z, x_f.w + y_f.w};
}
__device__ __forceinline__ float4 f4_sub(const float4& x_f, const float4& y_f) {
  return {x_f.x - y_f.x, x_f.y - y_f.y, x_f.z - y_f.z, x_f.w - y_f.w};
}
__device__ __forceinline__ float4 fast_sig_vec4(const float4& x_vec4) {
  const float sig_x = __frcp_rn(1.0f + __expf(-x_vec4.x));
  const float sig_y = __frcp_rn(1.0f + __expf(-x_vec4.y));
  const float sig_z = __frcp_rn(1.0f + __expf(-x_vec4.z));
  const float sig_w = __frcp_rn(1.0f + __expf(-x_vec4.w));
  return {sig_x, sig_y, sig_z, sig_w};
}
__device__ __forceinline__ float4
load_and_cast_float4(const bfloat16x4_t* x_vec4_ptr) {
  bfloat16x4_t x_vec4 = *x_vec4_ptr;
  return {
      static_cast<float>(x_vec4.x),
      static_cast<float>(x_vec4.y),
      static_cast<float>(x_vec4.z),
      static_cast<float>(x_vec4.w),
  };
}
__device__ __forceinline__ void cast_and_store_bf16x4(bfloat16x4_t* dst_ptr,
                                                      const float4& x_vec4) {
  *dst_ptr = {static_cast<__nv_bfloat16>(x_vec4.x),
              static_cast<__nv_bfloat16>(x_vec4.y),
              static_cast<__nv_bfloat16>(x_vec4.z),
              static_cast<__nv_bfloat16>(x_vec4.w)};
}
__device__ __forceinline__ float mreduce_f4(const float4& x_f4,
                                            const float4& y_f4) {
  float x_m = x_f4.x * y_f4.x;
  float y_m = x_f4.y * y_f4.y;
  float z_m = x_f4.z * y_f4.z;
  float w_m = x_f4.w * y_f4.w;
  return {x_m + y_m + z_m + w_m};
}

template <int thread_per_block>
__global__ void SwigluProbsGradKernelVec4(
    const BFloat16* o1,           // [seq_len*topk, moe_intermediate_size*2]
    const BFloat16* do2_s,        // [seq_len*topk, moe_intermediate_size]
    const float* unzipped_probs,  // [seq_len*topk, 1]
    BFloat16* do1,                // [seq_len*topk, moe_intermediate_size*2]
    float* probs_grad,            // [seq_len*topk, 1]
    BFloat16* o2_s,               // [seq_len*topk, moe_intermediate_size]
    int moe_intermediate_size) {
  constexpr int numel_per_thread = 4;
  constexpr int k_warp_size = 32;
  const int row_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const BFloat16* o1_row = o1 + row_idx * moe_intermediate_size * 2;
  const BFloat16* do2_s_row = do2_s + row_idx * moe_intermediate_size;
  const bfloat16x4_t* o1_row_left_half_vec4 =
      reinterpret_cast<const bfloat16x4_t*>(o1_row);
  const bfloat16x4_t* do2_s_row_vec4 =
      reinterpret_cast<const bfloat16x4_t*>(do2_s_row);
  const bfloat16x4_t* o1_row_right_half_vec4 =
      reinterpret_cast<const bfloat16x4_t*>(o1_row + moe_intermediate_size);
  BFloat16* do1_row = do1 + row_idx * moe_intermediate_size * 2;
  BFloat16* o2s_row = o2_s + row_idx * moe_intermediate_size;
  bfloat16x4_t* do1_row_vec4 = reinterpret_cast<bfloat16x4_t*>(do1_row);
  bfloat16x4_t* o2s_row_vec4 = reinterpret_cast<bfloat16x4_t*>(o2s_row);

  float prob = unzipped_probs[row_idx];
  __shared__ float sum_buffer[thread_per_block];

  float local_probs_grad = 0.0f;

  const int vec_numel = moe_intermediate_size / numel_per_thread;
  for (int i = tid; i < vec_numel; i += blockDim.x) {
    float4 lhs_vec4 = load_and_cast_float4(o1_row_left_half_vec4 + i);
    float4 rhs_vec4 = load_and_cast_float4(o1_row_right_half_vec4 + i);
    float4 do2_s_val_vec4 = load_and_cast_float4(do2_s_row_vec4 + i);
    float4 sig_vec4 = fast_sig_vec4(lhs_vec4);
    float4 tmp_vec4 = f4_prod(sig_vec4, lhs_vec4);
    float4 o2_val_vec4 = f4_prod(tmp_vec4, rhs_vec4);
    float4 o2s_val_vec4 = f4_prod(o2_val_vec4, prob);
    float4 do2_val_vec4 = f4_prod(do2_s_val_vec4, prob);
    float4 x0_grad_vec4 = f4_prod(
        do2_val_vec4,
        f4_prod(rhs_vec4,
                f4_prod(sig_vec4, (f4_sub(f4_add(lhs_vec4, 1.0f), tmp_vec4)))));
    float4 x1_grad_vec4 = f4_prod(do2_val_vec4, tmp_vec4);
    cast_and_store_bf16x4(do1_row_vec4 + i, x0_grad_vec4);
    cast_and_store_bf16x4(do1_row_vec4 + i + vec_numel, x1_grad_vec4);
    cast_and_store_bf16x4(o2s_row_vec4 + i, o2s_val_vec4);
    local_probs_grad += mreduce_f4(do2_s_val_vec4, o2_val_vec4);
  }

  sum_buffer[tid] = local_probs_grad;
  __syncthreads();

#pragma unroll
  for (int stride = blockDim.x / 2; stride >= k_warp_size; stride >>= 1) {
    if (tid < stride) {
      sum_buffer[tid] += sum_buffer[tid + stride];
    }
    __syncthreads();
  }

  if (tid < k_warp_size) {
    local_probs_grad = sum_buffer[tid];
#pragma unroll
    for (int offset = k_warp_size / 2; offset > 0; offset >>= 1) {
      local_probs_grad +=
          __shfl_down_sync(0xFFFFFFFF, local_probs_grad, offset);
    }
  }

  if (tid == 0) {
    probs_grad[row_idx] = local_probs_grad;
  }
}

template <typename T, typename Context>
void FusedSwigluWeightedBwdKernel(const Context& dev_ctx,
                                  const DenseTensor& o1,
                                  const DenseTensor& do2_s,
                                  const DenseTensor& unzipped_probs,
                                  DenseTensor* do1,
                                  DenseTensor* probs_grad,
                                  DenseTensor* o2_s) {
  auto o1_dims = o1.dims();
  int o1_outer_dim = 1;
  for (int i = 0; i < o1_dims.size() - 1; i++) {
    o1_outer_dim *= o1_dims[i];
  }

  const int moe_intermediate_size_2 = o1_dims[o1_dims.size() - 1];
  const int moe_intermediate_size = moe_intermediate_size_2 / 2;

  do1->Resize(o1.dims());
  dev_ctx.template Alloc<T>(do1);

  probs_grad->Resize({unzipped_probs.dims()});
  dev_ctx.template Alloc<float>(probs_grad);

  o2_s->Resize(do2_s.dims());
  dev_ctx.template Alloc<T>(o2_s);

  const BFloat16* o1_ptr = reinterpret_cast<const BFloat16*>(o1.data<T>());
  const BFloat16* do2_s_ptr =
      reinterpret_cast<const BFloat16*>(do2_s.data<T>());
  const float* unzipped_probs_ptr = unzipped_probs.data<float>();
  BFloat16* do1_ptr = reinterpret_cast<BFloat16*>(do1->data<T>());
  float* probs_grad_ptr = probs_grad->data<float>();
  BFloat16* o2_s_ptr = reinterpret_cast<BFloat16*>(o2_s->data<T>());

  constexpr int block_size = 256;

  if (moe_intermediate_size % 4 != 0) {
    SwigluProbsGradKernel<block_size>
        <<<o1_outer_dim, block_size, 0, dev_ctx.stream()>>>(
            o1_ptr,
            do2_s_ptr,
            unzipped_probs_ptr,
            do1_ptr,
            probs_grad_ptr,
            o2_s_ptr,
            moe_intermediate_size);
  } else {
    SwigluProbsGradKernelVec4<block_size>
        <<<o1_outer_dim, block_size, 0, dev_ctx.stream()>>>(
            o1_ptr,
            do2_s_ptr,
            unzipped_probs_ptr,
            do1_ptr,
            probs_grad_ptr,
            o2_s_ptr,
            moe_intermediate_size);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_swiglu_weighted_bwd,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedSwigluWeightedBwdKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BFLOAT16);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::BFLOAT16);
}
