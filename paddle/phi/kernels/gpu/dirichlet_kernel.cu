

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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/dirichlet_kernel_impl.h"

#ifdef PADDLE_WITH_CUDA
#include <curand_kernel.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hiprand_kernel.h>
#endif

#if defined(PADDLE_WITH_CUDA)
using COMPAT_RANDSTATEPHILOX4_32_10_T = curandStatePhilox4_32_10_t;
#define COMPAT_RAND_INIT curand_init
#define COMPAT_RAND_UNIFORM curand_uniform
#define COMPAT_RAND_NORMAL curand_normal
#elif defined(PADDLE_WITH_HIP)
using COMPAT_RANDSTATEPHILOX4_32_10_T = hiprandStatePhilox4_32_10_t;
#define COMPAT_RAND_INIT hiprand_init
#define COMPAT_RAND_UNIFORM hiprand_uniform
#define COMPAT_RAND_NORMAL hiprand_normal
#endif

namespace phi {
template <typename T>
struct GammaCUDAFunctor {
  GammaCUDAFunctor(const T* alpha, T* gamma, uint64_t seed, uint64_t offset)
      : alpha_(alpha), gamma_(gamma), seed_(seed), offset_(offset) {}

  DEVICE void operator()(int64_t index) {
    // curand initialization
    COMPAT_RANDSTATEPHILOX4_32_10_T state;
    COMPAT_RAND_INIT(
        /*seed=*/seed_, /*subsequence=*/index, /*offset=*/offset_, &state);

    // sample
    auto uniform_lambda = [&state]() { return COMPAT_RAND_UNIFORM(&state); };
    BaseSampler<T, decltype(uniform_lambda)> standard_uniform(uniform_lambda);
    auto normal_lambda = [&state]() { return COMPAT_RAND_NORMAL(&state); };
    BaseSampler<T, decltype(normal_lambda)> standard_normal(normal_lambda);

    auto sample =
        sample_gamma<T, T, decltype(uniform_lambda), decltype(normal_lambda)>(
            alpha_[index], standard_uniform, standard_normal);
    gamma_[index] = std::max(std::numeric_limits<T>::min(), sample);
  }

  const T* alpha_;
  T* gamma_;
  const uint64_t seed_;
  const uint64_t offset_;
};

template <typename T>
struct DirichletSampler<GPUContext, T> {
  void operator()(const GPUContext& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out) {
    auto p_gen = dev_ctx.GetGenerator();
    auto seed_and_offset = p_gen->IncrementOffset(10);  // hard-coded offset
    auto seed = seed_and_offset.first;
    auto offset = seed_and_offset.second;

    // sample from K gamma distributions, where K=alpha.numel()
    DenseTensor gamma_samples;
    gamma_samples.Resize(alpha.dims());
    dev_ctx.template Alloc<T>(&gamma_samples);

    GammaCUDAFunctor<T> gamma_functor(
        alpha.data<T>(), gamma_samples.data<T>(), seed, offset);
    funcs::ForRange<GPUContext> for_range(dev_ctx, out->numel());
    for_range(gamma_functor);

    // normalize them into a simplex, along the last axis
    DenseTensor gamma_sum;
    auto new_shape = gamma_samples.dims();
    new_shape[new_shape.size() - 1] = 1;
    gamma_sum.Resize(new_shape);
    dev_ctx.template Alloc<T>(&gamma_sum);

    funcs::ReduceKernelImpl<GPUContext, T, T, funcs::SumFunctor>(
        dev_ctx,
        gamma_samples,
        &gamma_sum,
        {new_shape.size() - 1},
        true,
        false);
    funcs::ElementwiseCompute<funcs::DivideFunctor<T>, T, T>(
        dev_ctx, gamma_samples, gamma_sum, -1, funcs::DivideFunctor<T>(), out);
  }
};
}  // namespace phi

PD_REGISTER_KERNEL(
    dirichlet, GPU, ALL_LAYOUT, phi::Dirichletkernel, float, double) {}
