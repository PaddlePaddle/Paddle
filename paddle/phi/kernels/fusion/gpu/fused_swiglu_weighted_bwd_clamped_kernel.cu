// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include <limits>
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

// ---------------------------------------------------------------------------
// Scalar (non-vectorized) kernel
// ---------------------------------------------------------------------------
template <int thread_per_block>
__global__ void SwigluProbsGradClampedKernel(const BFloat16* o1,
                                             const BFloat16* do2_s,
                                             const float* unzipped_probs,
                                             BFloat16* do1,
                                             float* probs_grad,
                                             BFloat16* o2_s,
                                             int64_t moe_intermediate_size,
                                             float clamp_value) {
  const int64_t row_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const BFloat16* o1_row = o1 + row_idx * moe_intermediate_size * 2;
  const BFloat16* do2_s_row = do2_s + row_idx * moe_intermediate_size;
  BFloat16* do1_row = do1 + row_idx * moe_intermediate_size * 2;
  BFloat16* o2s_row = o2_s + row_idx * moe_intermediate_size;
  float prob = unzipped_probs[row_idx];
  __shared__ float sum_buffer[thread_per_block];
  float local_probs_grad = 0.0f;

  for (int64_t i = tid; i < moe_intermediate_size; i += blockDim.x) {
    float lhs_raw = static_cast<float>(o1_row[i]);
    float rhs_raw = static_cast<float>(o1_row[i + moe_intermediate_size]);
    float lhs = fminf(lhs_raw, clamp_value);
    float rhs = fmaxf(fminf(rhs_raw, clamp_value), -clamp_value);
    float g_mask = (lhs_raw <= clamp_value) ? 1.0f : 0.0f;
    float v_mask =
        (rhs_raw <= clamp_value && rhs_raw >= -clamp_value) ? 1.0f : 0.0f;
    float sig = 1.0f / (1.0f + expf(-lhs));
    float tmp = sig * lhs;
    float o2_val = tmp * rhs;
    float do2_val = static_cast<float>(do2_s_row[i]) * prob;
    do1_row[i] = BFloat16(do2_val * rhs * sig * (1.0f + lhs - tmp) * g_mask);
    do1_row[i + moe_intermediate_size] = BFloat16(do2_val * tmp * v_mask);
    o2s_row[i] = BFloat16(o2_val * prob);
    local_probs_grad += static_cast<float>(do2_s_row[i]) * o2_val;
  }

  sum_buffer[tid] = local_probs_grad;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sum_buffer[tid] += sum_buffer[tid + stride];
    __syncthreads();
  }
  if (tid == 0) probs_grad[row_idx] = sum_buffer[0];
}

// ---------------------------------------------------------------------------
// Vec4 helpers
// ---------------------------------------------------------------------------
typedef struct __align__(8) {
  __nv_bfloat16 x, y, z, w;
}
bfloat16x4_t;

__device__ __forceinline__ float4 f4_prod(const float4& a, const float4& b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
__device__ __forceinline__ float4 f4_prod(const float4& a, float b) {
  return {a.x * b, a.y * b, a.z * b, a.w * b};
}
__device__ __forceinline__ float4 f4_add(const float4& a, float b) {
  return {a.x + b, a.y + b, a.z + b, a.w + b};
}
__device__ __forceinline__ float4 f4_add(const float4& a, const float4& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
__device__ __forceinline__ float4 f4_sub(const float4& a, const float4& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
__device__ __forceinline__ float4 fast_sig_vec4(const float4& v) {
  return {__frcp_rn(1.0f + __expf(-v.x)),
          __frcp_rn(1.0f + __expf(-v.y)),
          __frcp_rn(1.0f + __expf(-v.z)),
          __frcp_rn(1.0f + __expf(-v.w))};
}
__device__ __forceinline__ float4 f4_clamp_max(const float4& v, float cv) {
  return {fminf(v.x, cv), fminf(v.y, cv), fminf(v.z, cv), fminf(v.w, cv)};
}
__device__ __forceinline__ float4 f4_clamp(const float4& v, float cv) {
  return {fmaxf(fminf(v.x, cv), -cv),
          fmaxf(fminf(v.y, cv), -cv),
          fmaxf(fminf(v.z, cv), -cv),
          fmaxf(fminf(v.w, cv), -cv)};
}
__device__ __forceinline__ float4 f4_le_mask(const float4& v, float cv) {
  return {v.x <= cv ? 1.0f : 0.0f,
          v.y <= cv ? 1.0f : 0.0f,
          v.z <= cv ? 1.0f : 0.0f,
          v.w <= cv ? 1.0f : 0.0f};
}
__device__ __forceinline__ float4 f4_band_mask(const float4& v, float cv) {
  return {(v.x <= cv && v.x >= -cv) ? 1.0f : 0.0f,
          (v.y <= cv && v.y >= -cv) ? 1.0f : 0.0f,
          (v.z <= cv && v.z >= -cv) ? 1.0f : 0.0f,
          (v.w <= cv && v.w >= -cv) ? 1.0f : 0.0f};
}
__device__ __forceinline__ float4 load_bf16x4(const bfloat16x4_t* p) {
  bfloat16x4_t v = *p;
  return {static_cast<float>(v.x),
          static_cast<float>(v.y),
          static_cast<float>(v.z),
          static_cast<float>(v.w)};
}
__device__ __forceinline__ void store_bf16x4(bfloat16x4_t* p, const float4& v) {
  *p = {static_cast<__nv_bfloat16>(v.x),
        static_cast<__nv_bfloat16>(v.y),
        static_cast<__nv_bfloat16>(v.z),
        static_cast<__nv_bfloat16>(v.w)};
}
__device__ __forceinline__ float mreduce_f4(const float4& a, const float4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// ---------------------------------------------------------------------------
// Vec4 (vectorized) kernel
// ---------------------------------------------------------------------------
template <int thread_per_block>
__global__ void SwigluProbsGradClampedKernelVec4(const BFloat16* o1,
                                                 const BFloat16* do2_s,
                                                 const float* unzipped_probs,
                                                 BFloat16* do1,
                                                 float* probs_grad,
                                                 BFloat16* o2_s,
                                                 int64_t moe_intermediate_size,
                                                 float clamp_value) {
  constexpr int numel_per_thread = 4;
  constexpr int k_warp_size = 32;
  const int64_t row_idx = blockIdx.x;
  const int64_t tid = threadIdx.x;
  const BFloat16* o1_row = o1 + row_idx * moe_intermediate_size * 2;
  const bfloat16x4_t* lhs_v4 = reinterpret_cast<const bfloat16x4_t*>(o1_row);
  const bfloat16x4_t* rhs_v4 =
      reinterpret_cast<const bfloat16x4_t*>(o1_row + moe_intermediate_size);
  const bfloat16x4_t* do2_v4 = reinterpret_cast<const bfloat16x4_t*>(
      do2_s + row_idx * moe_intermediate_size);
  BFloat16* do1_row = do1 + row_idx * moe_intermediate_size * 2;
  BFloat16* o2s_row = o2_s + row_idx * moe_intermediate_size;
  bfloat16x4_t* do1_v4 = reinterpret_cast<bfloat16x4_t*>(do1_row);
  bfloat16x4_t* o2s_v4 = reinterpret_cast<bfloat16x4_t*>(o2s_row);
  float prob = unzipped_probs[row_idx];
  __shared__ float sum_buffer[thread_per_block];
  float local_probs_grad = 0.0f;

  const int64_t vec_numel = moe_intermediate_size / numel_per_thread;
  for (int64_t i = tid; i < vec_numel; i += blockDim.x) {
    float4 lhs_raw = load_bf16x4(lhs_v4 + i);
    float4 rhs_raw = load_bf16x4(rhs_v4 + i);
    float4 do2_s_val = load_bf16x4(do2_v4 + i);
    float4 lhs = f4_clamp_max(lhs_raw, clamp_value);
    float4 rhs = f4_clamp(rhs_raw, clamp_value);
    float4 g_mask = f4_le_mask(lhs_raw, clamp_value);
    float4 v_mask = f4_band_mask(rhs_raw, clamp_value);
    float4 sig = fast_sig_vec4(lhs);
    float4 tmp = f4_prod(sig, lhs);
    float4 o2_val = f4_prod(tmp, rhs);
    float4 do2_val = f4_prod(do2_s_val, prob);
    float4 x0_grad = f4_prod(
        f4_prod(do2_val,
                f4_prod(rhs, f4_prod(sig, (f4_sub(f4_add(lhs, 1.0f), tmp))))),
        g_mask);
    float4 x1_grad = f4_prod(f4_prod(do2_val, tmp), v_mask);
    store_bf16x4(do1_v4 + i, x0_grad);
    store_bf16x4(do1_v4 + i + vec_numel, x1_grad);
    store_bf16x4(o2s_v4 + i, f4_prod(o2_val, prob));
    local_probs_grad += mreduce_f4(do2_s_val, o2_val);
  }

  sum_buffer[tid] = local_probs_grad;
  __syncthreads();
#pragma unroll
  for (int stride = blockDim.x / 2; stride >= k_warp_size; stride >>= 1) {
    if (tid < stride) sum_buffer[tid] += sum_buffer[tid + stride];
    __syncthreads();
  }
  if (tid < k_warp_size) {
    local_probs_grad = sum_buffer[tid];
#pragma unroll
    for (int offset = k_warp_size / 2; offset > 0; offset >>= 1)
      local_probs_grad +=
          __shfl_down_sync(0xFFFFFFFF, local_probs_grad, offset);
  }
  if (tid == 0) probs_grad[row_idx] = local_probs_grad;
}

// ---------------------------------------------------------------------------
// Host function
// ---------------------------------------------------------------------------
template <typename T, typename Context>
void FusedSwigluWeightedBwdClampedKernel(const Context& dev_ctx,
                                         const DenseTensor& o1,
                                         const DenseTensor& do2_s,
                                         const DenseTensor& unzipped_probs,
                                         double clamp_value,
                                         DenseTensor* do1,
                                         DenseTensor* probs_grad,
                                         DenseTensor* o2_s) {
  if (o1.numel() == 0) {
    do1->Resize(o1.dims());
    dev_ctx.template Alloc<T>(do1);
    probs_grad->Resize(unzipped_probs.dims());
    dev_ctx.template Alloc<float>(probs_grad);
    o2_s->Resize(do2_s.dims());
    dev_ctx.template Alloc<T>(o2_s);
    return;
  }

  auto o1_dims = o1.dims();
  int64_t o1_outer_dim = 1;
  for (int i = 0; i < o1_dims.size() - 1; i++) o1_outer_dim *= o1_dims[i];
  const int64_t moe_intermediate_size_2 = o1_dims[o1_dims.size() - 1];
  const int64_t moe_intermediate_size = moe_intermediate_size_2 / 2;
  PADDLE_ENFORCE_LE(moe_intermediate_size_2,
                    std::numeric_limits<int>::max(),
                    common::errors::InvalidArgument(
                        "The last dimension of o1 (%d) exceeds int32 limit.",
                        moe_intermediate_size_2));

  do1->Resize(o1.dims());
  dev_ctx.template Alloc<T>(do1);
  probs_grad->Resize({unzipped_probs.dims()});
  dev_ctx.template Alloc<float>(probs_grad);
  o2_s->Resize(do2_s.dims());
  dev_ctx.template Alloc<T>(o2_s);

  const BFloat16* o1_ptr = reinterpret_cast<const BFloat16*>(o1.data<T>());
  const BFloat16* do2_s_ptr =
      reinterpret_cast<const BFloat16*>(do2_s.data<T>());
  const float* probs_ptr = unzipped_probs.data<float>();
  BFloat16* do1_ptr = reinterpret_cast<BFloat16*>(do1->data<T>());
  float* pg_ptr = probs_grad->data<float>();
  BFloat16* o2s_ptr = reinterpret_cast<BFloat16*>(o2_s->data<T>());

  constexpr int block_size = 256;
  const float cv = static_cast<float>(clamp_value);
  if (moe_intermediate_size % 4 != 0) {
    SwigluProbsGradClampedKernel<block_size>
        <<<o1_outer_dim, block_size, 0, dev_ctx.stream()>>>(
            o1_ptr,
            do2_s_ptr,
            probs_ptr,
            do1_ptr,
            pg_ptr,
            o2s_ptr,
            moe_intermediate_size,
            cv);
  } else {
    SwigluProbsGradClampedKernelVec4<block_size>
        <<<o1_outer_dim, block_size, 0, dev_ctx.stream()>>>(
            o1_ptr,
            do2_s_ptr,
            probs_ptr,
            do1_ptr,
            pg_ptr,
            o2s_ptr,
            moe_intermediate_size,
            cv);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_swiglu_weighted_bwd_clamped,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedSwigluWeightedBwdClampedKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BFLOAT16);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::BFLOAT16);
}
