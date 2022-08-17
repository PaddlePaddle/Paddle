// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/p_norm_kernel.h"

namespace phi {

#define FULL_MASK 0xffffffff

template <typename T>
__global__ void DeviceReduceSumZeroWithSubtract(const T* x,
                                                const T* y,
                                                T* out,
                                                int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum_val += abs(static_cast<T>(static_cast<double>(x[i] - y[i]) != 0));
  }

  __syncthreads();
  sum_val = phi::funcs::blockReduceSum<T>(sum_val, FULL_MASK);
  if (threadIdx.x == 0) out[blockIdx.x] = sum_val;
}

template <typename T>
__global__ void DeviceReduceSumZeroFinal(const T* x, T* out, int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum_val += x[i];
  }

  __syncthreads();
  sum_val = phi::funcs::blockReduceSum<T>(sum_val, FULL_MASK);
  if (threadIdx.x == 0) out[blockIdx.x] = sum_val;
}

template <typename T>
__global__ void DeviceReduceMaxWithSubtract(const T* x,
                                            const T* y,
                                            T* out,
                                            int64_t N) {
  T max_val = -1e10f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    max_val = max(max_val, abs(x[i] - y[i]));
  }

  __syncthreads();
  max_val = phi::funcs::blockReduceMax<T>(max_val, FULL_MASK);
  if (threadIdx.x == 0) out[blockIdx.x] = max_val;
}

template <typename T>
__global__ void DeviceReduceMaxFinal(const T* x, T* out, int64_t N) {
  T max_val = -1e10f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    max_val = max(max_val, abs(x[i]));
  }
  __syncthreads();
  max_val = phi::funcs::blockReduceMax<T>(max_val, FULL_MASK);
  if (threadIdx.x == 0) out[blockIdx.x] = max_val;
}

template <typename T>
__global__ void DeviceReduceMinWithSubtract(const T* x,
                                            const T* y,
                                            T* out,
                                            int64_t N) {
  T min_val = 1e10f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    min_val = min(min_val, abs(x[i] - y[i]));
  }

  __syncthreads();
  min_val = phi::funcs::blockReduceMin(min_val, FULL_MASK);
  if (threadIdx.x == 0) out[blockIdx.x] = min_val;
}

template <typename T>
__global__ void DeviceReduceMinFinal(const T* x, T* out, int64_t N) {
  T min_val = 1e10f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    min_val = min(min_val, abs(x[i]));
  }
  __syncthreads();
  min_val = phi::funcs::blockReduceMin(min_val, FULL_MASK);
  if (threadIdx.x == 0) out[blockIdx.x] = min_val;
}

template <typename T>
__global__ void DeviceReduceSumPOrderWithSubtract(
    const T* x, const T* y, T* out, T p_order, int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum_val += static_cast<T>(pow(abs(x[i] - y[i]), p_order));
  }

  __syncthreads();
  sum_val = phi::funcs::blockReduceSum<T>(sum_val, FULL_MASK);
  if (threadIdx.x == 0) out[blockIdx.x] = sum_val;
}

template <typename T>
__global__ void DeviceReduceSumPOrderFinal(const T* x,
                                           T* out,
                                           T p_order,
                                           int64_t N) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum_val += x[i];
  }

  __syncthreads();
  sum_val = phi::funcs::blockReduceSum<T>(sum_val, FULL_MASK);
  if (threadIdx.x == 0) out[blockIdx.x] = pow(sum_val, (1 / p_order));
}

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
      DeviceReduceSumZeroWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);
      DeviceReduceSumZeroFinal<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, config.block_per_grid.x);

    } else if (p == INFINITY) {
      DeviceReduceMaxWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);
      DeviceReduceMaxFinal<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, config.block_per_grid.x);

    } else if (p == -INFINITY) {
      DeviceReduceMinWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);
      DeviceReduceMinFinal<T><<<1, config.thread_per_block.x, 0, stream>>>(
          i_ptr, o_ptr, config.block_per_grid.x);

    } else {
      T p_order = static_cast<T>(p);
      DeviceReduceSumPOrderWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, p_order, n);
      DeviceReduceSumPOrderFinal<T>
          <<<1, config.thread_per_block.x, 0, stream>>>(
              i_ptr, o_ptr, p_order, config.block_per_grid.x);
    }

  } else {
    auto t = Subtract<T, Context>(dev_ctx, x, y);
    PNormKernel<T, Context>(dev_ctx, t, p, -1, 1e-12, false, true, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(dist, GPU, ALL_LAYOUT, phi::DistKernel, float, double) {}
