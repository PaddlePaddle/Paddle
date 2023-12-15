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

#include <algorithm>

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/dist_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/legacy/reduce_max_kernel.h"
#include "paddle/phi/kernels/p_norm_kernel.h"
#include "paddle/phi/kernels/reduce_min_kernel.h"

namespace phi {

#define FULL_MASK 0xffffffff

template <typename Tx, typename Ty = Tx>
struct ZeroOrderFunctor {
 public:
  HOSTDEVICE explicit inline ZeroOrderFunctor() {}
  HOSTDEVICE inline Ty operator()(const Tx& x, const Tx& y) const {
    return static_cast<Ty>(x != y);
  }
};

template <typename Tx, typename Ty = Tx>
struct OtherOrderFunctor {
  HOSTDEVICE explicit inline OtherOrderFunctor(const Ty& p_order)
      : p_order_(p_order) {}

  HOSTDEVICE inline Ty operator()(const Tx& x, const Tx& y) const {
    return static_cast<Ty>(
        pow(abs(static_cast<Ty>(x) - static_cast<Ty>(y)), p_order_));
  }

 private:
  Ty p_order_;
};

template <typename Tx, typename Ty = Tx>
struct PowFunctor {
  HOSTDEVICE explicit inline PowFunctor(const Ty& p_order)
      : p_order_(p_order) {}
  HOSTDEVICE inline Tx operator()(const Tx x) const {
    return static_cast<Tx>(pow(static_cast<Ty>(x), p_order_));
  }
  Ty p_order_;
};

template <typename T, typename Functor>
__global__ void ReduceSumWithSubtract(
    const T* x, const T* y, T* out, int64_t N, Functor func) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  MT sum_val(0.0);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum_val += func(x[i], y[i]);
  }

  __syncthreads();
  sum_val = phi::funcs::BlockReduceSum<MT>(sum_val, FULL_MASK);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = static_cast<T>(sum_val);
  }
}

template <typename T>
__global__ void ReduceMaxWithSubtract(const T* x,
                                      const T* y,
                                      T* out,
                                      int64_t N) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  MT max_val = std::numeric_limits<MT>::min();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    max_val = max(max_val, abs(static_cast<MT>(x[i]) - static_cast<MT>(y[i])));
  }

  __syncthreads();
  max_val = phi::funcs::BlockReduceMax<MT>(max_val, FULL_MASK);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = static_cast<T>(max_val);
  }
}

template <typename T>
__global__ void ReduceMinWithSubtract(const T* x,
                                      const T* y,
                                      T* out,
                                      int64_t N) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  MT min_val = std::numeric_limits<MT>::max();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    min_val = min(min_val, abs(static_cast<MT>(x[i]) - static_cast<MT>(y[i])));
  }

  __syncthreads();
  min_val = phi::funcs::BlockReduceMin<MT>(min_val, FULL_MASK);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = static_cast<T>(min_val);
  }
}

template <typename T, typename Context>
void DistKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                float p,
                DenseTensor* out) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  DenseTensor intermediate;
  const T* x_ptr = x.data<T>();
  const T* y_ptr = y.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  auto xdim = x.dims();
  if (xdim == y.dims()) {  // same shape
    auto n = x.numel();
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n);
    intermediate.Resize(common::make_ddim({config.block_per_grid.x}));
    T* i_ptr = dev_ctx.template Alloc<T>(&intermediate);

    std::vector<int64_t> axis_dims = {static_cast<int64_t>(-1)};
    std::vector<int> reduce_axis =
        funcs::details::GetReduceDim(axis_dims, xdim.size(), true);

    if (p == 0) {
      ReduceSumWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n, ZeroOrderFunctor<T, MT>());
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<MT>>(
          dev_ctx, intermediate, out, kps::IdentityFunctor<MT>(), reduce_axis);
    } else if (p == INFINITY) {
      ReduceMaxWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);
      phi::MaxRawKernel<T, Context>(
          dev_ctx, intermediate, reduce_axis, true, true, out);

    } else if (p == -INFINITY) {
      ReduceMinWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n);

      phi::MinRawKernel<T, Context>(
          dev_ctx, intermediate, reduce_axis, true, true, out);

    } else {
      MT p_order = static_cast<MT>(p);
      ReduceSumWithSubtract<T>
          <<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
              x_ptr, y_ptr, i_ptr, n, OtherOrderFunctor<T, MT>(p_order));
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<MT>>(
          dev_ctx, intermediate, out, kps::IdentityFunctor<MT>(), reduce_axis);

      const DenseTensor* tmp_norm = out;
      std::vector<const DenseTensor*> ins = {tmp_norm};
      std::vector<DenseTensor*> outs = {out};
      MT p_order_ = static_cast<MT>(static_cast<MT>(1.) / p_order);
      phi::funcs::ElementwiseKernel<T>(
          dev_ctx, ins, &outs, PowFunctor<T, MT>(p_order_));
    }

  } else {
    auto t = Subtract<T, Context>(dev_ctx, x, y);
    PNormKernel<T, Context>(dev_ctx, t, p, -1, 1e-12, false, true, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(dist,
                   GPU,
                   ALL_LAYOUT,
                   phi::DistKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
