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
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/p_norm_kernel.h"

namespace phi {

#define FULL_MASK 0xffffffff

template <typename T>
struct ZeroOrderFunctor {
 public:
  __device__ T operator()(const T& x, const T& y) const {
    return static_cast<T>((x - y) != 0);
  }
};

template <typename T>
struct OtherOrderFunctor {
  explicit OtherOrderFunctor(const T& p_order) : p_order_(p_order) {}
  __device__ T operator()(const T& x, const T& y) const {
    return static_cast<T>(pow(abs(x - y), p_order_));
  }

 private:
  T p_order_;
};

template <typename T>
struct PowFunctor {
  explicit PowFunctor(const T& p_order) : p_order_(p_order) {}
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(pow(x, p_order_));
  }
  T p_order_;
};

template <typename T, typename Functor>
__global__ void ReduceSumWithSubtract(
    const T* x, const T* y, T* out, int64_t N, Functor func) {
  T sum_val = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum_val += func(x[i], y[i]);
  }

  __syncthreads();
  sum_val = phi::funcs::blockReduceSum<T>(sum_val, FULL_MASK);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum_val;
  }
}

template <typename T>
__global__ void ReduceMaxWithSubtract(const T* x,
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
  if (threadIdx.x == 0) {
    out[blockIdx.x] = max_val;
  }
}

template <typename T>
__global__ void ReduceMinWithSubtract(const T* x,
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
  if (threadIdx.x == 0) {
    out[blockIdx.x] = min_val;
  }
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

  auto xdim = x.dims();
  if (xdim == y.dims()) {  // same shape
    auto n = x.numel();
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n);
    intermediate.Resize(phi::make_ddim({config.block_per_grid.x}));
    T* i_ptr = dev_ctx.template Alloc<T>(&intermediate);

    std::vector<int64_t> axis_dims = {static_cast<int64_t>(-1)};
    std::vector<int> reduce_axis =
        funcs::details::GetReduceDim(axis_dims, xdim.size(), true);

    if (p == 0) {
      ReduceSumWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n, ZeroOrderFunctor<T>());
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dev_ctx, intermediate, out, kps::IdentityFunctor<T>(), reduce_axis);

    } else if (p == INFINITY) {
      ReduceMaxWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);
      phi::funcs::ReduceKernel<T, T, kps::MaxFunctor, kps::IdentityFunctor<T>>(
          dev_ctx, intermediate, out, kps::IdentityFunctor<T>(), reduce_axis);

    } else if (p == -INFINITY) {
      ReduceMinWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);

      phi::funcs::ReduceKernel<T, T, kps::MinFunctor, kps::IdentityFunctor<T>>(
          dev_ctx, intermediate, out, kps::IdentityFunctor<T>(), reduce_axis);

    } else {
      T p_order = static_cast<T>(p);
      ReduceSumWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n, OtherOrderFunctor<T>(p_order));
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dev_ctx, intermediate, out, kps::IdentityFunctor<T>(), reduce_axis);

      const DenseTensor* tmp_norm = out;
      std::vector<const DenseTensor*> ins = {tmp_norm};
      std::vector<DenseTensor*> outs = {out};
      T p_order_ = static_cast<T>(1. / p_order);
      phi::funcs::ElementwiseKernel<T>(
          dev_ctx, ins, &outs, PowFunctor<T>(p_order_));
    }

  } else {
    auto t = Subtract<T, Context>(dev_ctx, x, y);
    PNormKernel<T, Context>(dev_ctx, t, p, -1, 1e-12, false, true, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(dist, GPU, ALL_LAYOUT, phi::DistKernel, float, double) {}
