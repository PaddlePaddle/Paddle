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

#include "paddle/phi/kernels/std_var_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/kernel_utils.h"
#include "paddle/phi/kernels/full_kernel.h"

#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/gpu/reduce.h"

#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define C10_HOST __host__

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/pair.h>
#else
#include <cmath>
#define device_sqrt std::sqrt
#endif

namespace phi {

#if defined(USE_ROCM)
#include <math.h>
template <typename scalar_t>
static __forceinline__ __device__ scalar_t device_sqrt(scalar_t val);

template <>
__forceinline__ __device__ float device_sqrt(float val) {
  return ::sqrtf(val);
}

template <>
__forceinline__ __device__ double device_sqrt(double val) {
  return ::sqrt(val);
}
#else
template <typename scalar_t>
__forceinline__ __device__ double device_sqrt(scalar_t val) {
  return std::sqrt(val);
}
#endif

template <typename T>
C10_DEVICE __forceinline__ T WARP_SHFL_DOWN(T value,
                                            unsigned int delta,
                                            int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#ifndef __HIPCC__
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;

  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0) {}

  C10_HOST_DEVICE WelfordData(scalar_t mean,
                              scalar_t m2,
                              index_t n,
                              scalar_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

template <typename scalar_t,
          typename acc_scalar_t,
          typename index_t,
          typename res_t>
struct WelfordOps {
  acc_scalar_t correction;
  bool take_sqrt;

 public:
  using acc_t = WelfordData<acc_scalar_t, index_t>;
  inline C10_DEVICE acc_t compute(acc_t acc, scalar_t data) const {
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    acc_scalar_t delta = static_cast<acc_scalar_t>(data) - acc.mean;
    acc_scalar_t new_mean = acc.mean + delta / new_nf;
    acc_scalar_t new_delta = static_cast<acc_scalar_t>(data) - new_mean;
    return {
        new_mean,
        acc.m2 + delta * new_delta,
        new_n,
        new_nf,
    };
  }
  inline C10_DEVICE acc_t reduce(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    acc_scalar_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {a.mean + delta * nb_over_n,
            a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
            -1,
            new_count};
  }
  inline C10_DEVICE res_t post_process(acc_t acc) const {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = static_cast<scalar_t>(acc.m2 / divisor);
    const auto var_sqrt =
        static_cast<scalar_t>(device_sqrt(static_cast<acc_scalar_t>(var)));
    res_t results(take_sqrt ? var_sqrt : var, mean);
    return results;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t shfl_sync(unsigned int mask,
                                    acc_t acc,
                                    int offset) const {
    return {WARP_SHFL_DOWN(acc.mean, offset),
            WARP_SHFL_DOWN(acc.m2, offset),
            WARP_SHFL_DOWN(acc.n, offset),
            WARP_SHFL_DOWN(acc.nf, offset)};
  }
#endif
  C10_HOST_DEVICE WelfordOps(acc_scalar_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

template <typename T, typename Context>
void Std_VarKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<int64_t>& axis,
                   bool keepdim,
                   double correction,
                   bool take_sqrt,
                   DenseTensor* out) {
  if (x.numel() == 0) {
    phi::Full<T, Context>(dev_ctx,
                          phi::IntArray(common::vectorize(out->dims())),
                          static_cast<T>(NAN),
                          out);
    return;
  }

  dev_ctx.template Alloc<T>(out);

  int64_t ndim = x.dims().size();
  std::vector<int32_t> axis32(axis.begin(), axis.end());
  auto positive_reduce_dims = ConvertToPositiveDims(axis32, ndim);
  auto mask = MakeDimMask(positive_reduce_dims, ndim);
  auto viewed_result = ReviewReduceResult(x, *(out), ndim, mask);

  DenseTensorIteratorConfig dense_iter_config;
  dense_iter_config.is_reduction(true);
  dense_iter_config.add_output(viewed_result);
  dense_iter_config.add_const_input(x);
  DenseTensorIterator iter = dense_iter_config.build();

  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  using ops_t = WelfordOps<T, AccT, int32_t, thrust::pair<T, T>>;
  ops_t ops(static_cast<AccT>(correction), take_sqrt);

  GPUReduceScheduler<T, T, 2>(dev_ctx, iter, ops, typename ops_t::acc_t{});
}

template <typename T, typename Context>
void VarKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  Std_VarKernel<T, Context>(dev_ctx, x, axis, keepdim, correction, false, out);
}

template <typename T, typename Context>
void StdKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  Std_VarKernel<T, Context>(dev_ctx, x, axis, keepdim, correction, true, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(var,
                   GPU,
                   ALL_LAYOUT,
                   phi::VarKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
PD_REGISTER_KERNEL(std,
                   GPU,
                   ALL_LAYOUT,
                   phi::StdKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
