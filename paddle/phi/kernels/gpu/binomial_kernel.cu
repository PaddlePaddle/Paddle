/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef __NVCC__
#include <curand_kernel.h>
#endif
#ifdef __HIPCC__
#include <hiprand_kernel.h>
#endif

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/binomial_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

__device__ __constant__ float kTailValues[] = {0.0810614667953272,
                                               0.0413406959554092,
                                               0.0276779256849983,
                                               0.02079067210376509,
                                               0.0166446911898211,
                                               0.0138761288230707,
                                               0.0118967099458917,
                                               0.0104112652619720,
                                               0.00925546218271273,
                                               0.00833056343336287};

template <typename T>
__device__ T stirling_approx_tail(int64_t k) {
  if (k <= 9) {
    return static_cast<T>(kTailValues[static_cast<size_t>(k)]);
  }
  T kp1sq = (k + 1) * (k + 1);
  return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
}

template <typename T>
__device__ int64_t btrs(
    const T n, const T p, int64_t idx, unsigned int seed, unsigned int offset) {
  int64_t k;
  T U, V, us;

#ifdef __NVCC__
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);
#elif __HIPCC__
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, offset, &state);
#endif

  const T stddev = std::sqrt(n * p * (1 - p));

  const T b = 1.15 + 2.53 * stddev;
  const T a = -0.0873 + 0.0248 * b + 0.01 * p;
  const T c = n * p + 0.5;
  const T v_r = 0.92 - 4.2 / b;
  const T r = p / (1 - p);

  const T alpha = (2.83 + 5.1 / b) * stddev;
  const T m = std::floor((n + 1) * p);

  while (1) {
#ifdef __NVCC__
    U = static_cast<T>(curand_uniform(&state)) - 0.5;
    V = static_cast<T>(curand_uniform(&state));
#elif __HIPCC__
    U = static_cast<T>(hiprand_uniform(&state)) - 0.5;
    V = static_cast<T>(hiprand_uniform(&state));
#endif

    us = 0.5 - std::abs(U);
    k = static_cast<int64_t>(std::floor((2 * a / us + b) * U + c));

    if (k < 0 || k > n) {
      continue;
    }
    if (us >= 0.07 && V <= v_r) {
      return k;
    }

    V = std::log(V * alpha / (a / (us * us) + b));
    T upperbound =
        ((m + 0.5) * std::log((m + 1) / (r * (n - m + 1))) +
         (n + 1) * std::log((n - m + 1) / (n - k + 1)) +
         (k + 0.5) * std::log(r * (n - k + 1) / (k + 1)) +
         stirling_approx_tail<T>(m) + stirling_approx_tail<T>(n - m) -
         stirling_approx_tail<T>(k) - stirling_approx_tail<T>(n - k));

    if (V <= upperbound) {
      return k;
    }
  }
}

template <typename T>
__device__ int64_t binomial_inversion(
    const T n, const T p, int64_t idx, unsigned int seed, unsigned int offset) {
  T unif;
  T geom_sum = 0.0;
  int64_t num_geom = 0;
  T logprob = std::log1p(-p);

#ifdef __NVCC__
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);
#elif __HIPCC__
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, offset, &state);
#endif

  while (1) {
#ifdef __NVCC__
    unif = static_cast<T>(curand_uniform(&state));
#elif __HIPCC__
    unif = static_cast<T>(hiprand_uniform(&state));
#endif
    T geom = std::ceil(std::log(unif) / logprob);
    geom_sum += geom;
    if (geom_sum > n) {
      break;
    }
    num_geom = num_geom + 1;
  }
  return num_geom;
}

template <typename T>
__global__ void BinomialSampling(const T* n,
                                 const T* p,
                                 int64_t* out,
                                 const int N,
                                 unsigned int seed,
                                 unsigned int offset) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  CUDA_KERNEL_LOOP_TYPE(idx, N, int64_t) {
    MT nt = static_cast<MT>(n[idx]);
    MT pt = static_cast<MT>(p[idx]);
    if (nt <= 0.0 || pt <= 0.0) {
      out[idx] = 0;
    } else if (pt >= 1.0) {
      out[idx] = static_cast<int64_t>(nt);
    } else if (pt <= 0.5) {
      if (nt * pt >= 10.0) {
        out[idx] = btrs<MT>(nt, pt, idx, seed, offset);
      } else {
        out[idx] = binomial_inversion<MT>(nt, pt, idx, seed, offset);
      }
    } else {
      MT qprob = 1.0 - pt;
      if (nt * qprob >= 10.0) {
        out[idx] =
            static_cast<int64_t>(nt) - btrs<MT>(nt, qprob, idx, seed, offset);
      } else {
        out[idx] = static_cast<int64_t>(nt) -
                   binomial_inversion<MT>(nt, qprob, idx, seed, offset);
      }
    }
  }
}

template <typename T, typename Context>
void BinomialKernel(const Context& ctx,
                    const DenseTensor& count,
                    const DenseTensor& prob,
                    DenseTensor* out) {
  const T* count_data = count.data<T>();
  const T* prob_data = prob.data<T>();
  int64_t* out_data = ctx.template Alloc<int64_t>(out);
  const int size = count.numel();
  const int kMaxBlockDim = 256;

  int block_size = std::min(kMaxBlockDim, ctx.GetMaxThreadsPerBlock());
  dim3 dim_block(block_size);
  dim3 dim_grid((size + block_size - 1) / block_size);
  phi::backends::gpu::LimitGridDim(ctx, &dim_grid);

  auto gen_cuda = ctx.GetGenerator();
  auto seed_offset = gen_cuda->IncrementOffset(20);
  uint64_t seed = seed_offset.first;
  uint64_t offset = seed_offset.second;
  BinomialSampling<T><<<dim_grid, dim_block>>>(
      count_data, prob_data, out_data, size, seed, offset);
}

}  // namespace phi

PD_REGISTER_KERNEL(binomial,
                   GPU,
                   ALL_LAYOUT,
                   phi::BinomialKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
