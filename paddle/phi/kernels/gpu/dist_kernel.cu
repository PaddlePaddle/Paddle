// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/dist_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"

#define FULL_MASK 0xffffffff
#define HALF_WARP_SIZE 16
#define WARP_SIZE 32

__device__ __forceinline__ float inline_abs(float x) { return abs(x); }
__device__ __forceinline__ double inline_abs(double x) { return abs(x); }
__device__ __forceinline__ float inline_pow(float base, float exponent) {
  return pow(base, exponent);
}
__device__ __forceinline__ double inline_pow(double base, double exponent) {
  return pow(base, exponent);
}

template <typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int offset = HALF_WARP_SIZE; offset > 0; offset /= 2) {
#ifdef __HIPCC__
    val += __shfl_down(val, offset);
#else
    val += __shfl_down_sync(FULL_MASK, val, offset);
#endif
  }
  return val;
}

template <typename T>
__forceinline__ __device__ T warpReduceMax(T val) {
#pragma unroll
  for (int offset = HALF_WARP_SIZE; offset > 0; offset /= 2) {
#ifdef __HIPCC__
    val = max(val, __shfl_xor(val, offset));
#else
    val = max(val, __shfl_xor_sync(FULL_MASK, val, offset));
#endif
  }
  return val;
}

template <typename T>
__forceinline__ __device__ T warpReduceMin(T val) {
#pragma unroll
  for (int offset = HALF_WARP_SIZE; offset > 0; offset /= 2) {
#ifdef __HIPCC__
    val = min(val, __shfl_xor(val, offset));
#else
    val = min(val, __shfl_xor_sync(FULL_MASK, val, offset));
#endif
  }
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  val = warpReduceSum<T>(val);
  __syncthreads();
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  int block_span = blockDim.x >> 5;
  val = (threadIdx.x < block_span) ? shared[lane] : static_cast<T>(0.0f);
  if (wid == 0) val = warpReduceSum<T>(val);
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  val = warpReduceMax<T>(val);
  __syncthreads();
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  int block_span = blockDim.x >> 5;
  val = (threadIdx.x < block_span) ? shared[lane] : -1e10f;
  if (wid == 0) val = warpReduceMax<T>(val);
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceMin(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  val = warpReduceMax<T>(val);
  __syncthreads();
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  int block_span = blockDim.x >> 5;
  val = (threadIdx.x < block_span) ? shared[lane] : 1e10f;
  if (wid == 0) val = warpReduceMax<T>(val);
  return val;
}

template <typename T>
__global__ void deviceReduceSumZero(const T* x, T* out, int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum_val += static_cast<T>(static_cast<double>(x[i]) != 0);
  __syncthreads();
  sum_val = blockReduceSum(sum_val);
  if (threadIdx.x == 0) out[blockIdx.x] = sum_val;
}

template <typename T>
__global__ void deviceReduceSumZeroWithSubstract(const T* x,
                                                 const T* y,
                                                 T* out,
                                                 int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum_val +=
        inline_abs(static_cast<T>(static_cast<double>(x[i] - y[i]) != 0));
  __syncthreads();
  sum_val = blockReduceSum(sum_val);
  if (threadIdx.x == 0) out[blockIdx.x] = sum_val;
}

template <typename T>
__global__ void deviceReduceSumZeroFinal(const T* x, T* out, int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum_val += x[i];
  __syncthreads();
  sum_val = blockReduceSum(sum_val);
  if (threadIdx.x == 0) out[blockIdx.x] = sum_val;
}

template <typename T>
__global__ void deviceReduceMax(const T* x, T* out, int64_t N) {
  T max_val = -1e10f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    max_val = max(max_val, inline_abs(x[i]));
  __syncthreads();
  max_val = blockReduceMax(max_val);
  if (threadIdx.x == 0) out[blockIdx.x] = max_val;
}

template <typename T>
__global__ void deviceReduceMaxWithSubstract(const T* x,
                                             const T* y,
                                             T* out,
                                             int64_t N) {
  T max_val = -1e10f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    max_val = max(max_val, inline_abs(x[i] - y[i]));
  __syncthreads();
  max_val = blockReduceMax(max_val);
  if (threadIdx.x == 0) out[blockIdx.x] = max_val;
}

template <typename T>
__global__ void deviceReduceMin(const T* x, T* out, int64_t N) {
  T min_val = 1e10f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    min_val = min(min_val, inline_abs(x[i]));
  __syncthreads();
  min_val = blockReduceMin(min_val);
  if (threadIdx.x == 0) out[blockIdx.x] = min_val;
}

template <typename T>
__global__ void deviceReduceMinWithSubstract(const T* x,
                                             const T* y,
                                             T* out,
                                             int64_t N) {
  T min_val = 1e10f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    min_val = min(min_val, inline_abs(x[i] - y[i]));
  __syncthreads();
  min_val = blockReduceMin(min_val);
  if (threadIdx.x == 0) out[blockIdx.x] = min_val;
}

template <typename T>
__global__ void deviceReduceSumOrder(const T* x, T* out, T p_order, int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum_val += static_cast<T>(inline_pow(inline_abs(x[i]), p_order));
  __syncthreads();
  sum_val = blockReduceSum(sum_val);
  if (threadIdx.x == 0) out[blockIdx.x] = sum_val;
}

template <typename T>
__global__ void deviceReduceSumOrderWithSubstract(
    const T* x, const T* y, T* out, T p_order, int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum_val += static_cast<T>(inline_pow(inline_abs(x[i] - y[i]), p_order));
  __syncthreads();
  sum_val = blockReduceSum(sum_val);
  if (threadIdx.x == 0) out[blockIdx.x] = sum_val;
}

template <typename T>
__global__ void deviceReduceSumOrderFinal(const T* x,
                                          T* out,
                                          T p_order,
                                          int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x)
    sum_val += x[i];
  __syncthreads();
  sum_val = blockReduceSum(sum_val);
  if (threadIdx.x == 0) out[blockIdx.x] = inline_pow(sum_val, (1 / p_order));
}

namespace phi {

template <typename T, typename Context>
void DistKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                float p,
                DenseTensor* out) {
  DenseTensor intermediate;
  const T* x_ptr = x.data<T>();
  const T* y_ptr = y.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  if (x.dims() == y.dims()) {  // same shape
    auto n = x.numel();
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n);
    intermediate.Resize(phi::make_ddim({config.block_per_grid.x}));
    T* i_ptr = dev_ctx.template Alloc<T>(&intermediate);

    if (p == 0) {
      deviceReduceSumZeroWithSubstract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);
      cudaStreamSynchronize(stream);
      deviceReduceSumZeroFinal<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, config.block_per_grid.x);

    } else if (p == INFINITY) {
      deviceReduceMaxWithSubstract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);
      cudaStreamSynchronize(stream);
      deviceReduceMax<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, config.block_per_grid.x);

    } else if (p == -INFINITY) {
      deviceReduceMinWithSubstract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);
      cudaStreamSynchronize(stream);
      deviceReduceMin<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, config.block_per_grid.x);

    } else {
      T p_order = static_cast<T>(p);
      deviceReduceSumOrderWithSubstract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, p_order, n);
      cudaStreamSynchronize(stream);
      deviceReduceSumOrderFinal<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, p_order, config.block_per_grid.x);
    }

  } else {
    DenseTensor t = Subtract<T, Context>(dev_ctx, x, y);
    const T* t_ptr = t.data<T>();
    auto n = t.numel();
    auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n);
    intermediate.Resize(phi::make_ddim({config.block_per_grid.x}));
    T* i_ptr = dev_ctx.template Alloc<T>(&intermediate);

    if (p == 0) {
      deviceReduceSumZero<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              t_ptr, i_ptr, n);
      cudaStreamSynchronize(stream);
      deviceReduceSumZeroFinal<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, config.block_per_grid.x);

    } else if (p == INFINITY) {
      deviceReduceMax<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              t_ptr, i_ptr, n);
      cudaStreamSynchronize(stream);
      deviceReduceMax<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, config.block_per_grid.x);

    } else if (p == -INFINITY) {
      deviceReduceMin<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              t_ptr, i_ptr, n);
      cudaStreamSynchronize(stream);
      deviceReduceMin<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, config.block_per_grid.x);
    } else {
      T p_order = static_cast<T>(p);
      deviceReduceSumOrder<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              t_ptr, i_ptr, p_order, n);
      cudaStreamSynchronize(stream);
      deviceReduceSumOrderFinal<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, p_order, config.block_per_grid.x);
    }

    cudaStreamSynchronize(stream);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(dist, GPU, ALL_LAYOUT, phi::DistKernel, float, double) {}
