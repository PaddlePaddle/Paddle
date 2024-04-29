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

#pragma once

#include <cmath>
#include <random>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/dirichlet_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

// ROCM hcc doesn't work well with using std:: in kernel functions
#if defined(PADDLE_WITH_CUDA)
#define COMPAT_EXP exp
#define COMPAT_CEIL ceil
#define COMPAT_FLOOR floor
#define COMPAT_LOG log
#define COMPAT_POW pow
#define COMPAT_SQRT sqrt
#define COMPAT_TAN tan
#define COMPAT_ABS abs
#define COMPAT_LOG1P log1p
#else
#define COMPAT_EXP std::exp
#define COMPAT_CEIL std::ceil
#define COMPAT_FLOOR std::floor
#define COMPAT_LOG std::log
#define COMPAT_POW std::pow
#define COMPAT_SQRT std::sqrt
#define COMPAT_TAN std::tan
#define COMPAT_ABS std::abs
#define COMPAT_LOG1P std::log1p
#endif

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

template <typename ScalarT, typename SamplerT>
struct BaseSampler {
  SamplerT sampler_;
  HOSTDEVICE BaseSampler(const SamplerT& sampler) : sampler_(sampler) {}
  HOSTDEVICE ScalarT sample() {
    // Sometimes convert float to float16/bfloat16
    return static_cast<ScalarT>(sampler_());
  }
};

template <typename Context, typename T>
struct GammaSampler {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out);
};

template <typename Context, typename T>
struct DirichletSampler {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out);
};

// `sample_gamma` is d from Numpy's distributions.c, and add support for
//  paddle data type and code style.
//  Source MIT licensed:
/* Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

template <typename ScalarT,
          typename AccscalarT,
          typename UniformSamplerT,
          typename NormalSamplerT>
HOSTDEVICE ScalarT
sample_gamma(ScalarT alpha,
             BaseSampler<AccscalarT, UniformSamplerT> standard_uniform,
             BaseSampler<AccscalarT, NormalSamplerT> standard_normal) {
  using MPTypeScalar = typename phi::dtype::MPTypeTrait<ScalarT>::Type;
  using MPTypeAccscalar = typename phi::dtype::MPTypeTrait<AccscalarT>::Type;

  MPTypeAccscalar mp_scale = static_cast<MPTypeAccscalar>(1.0f);
  MPTypeScalar mp_alpha = static_cast<MPTypeScalar>(alpha);

  // Boost alpha for higher acceptance probability.
  if (mp_alpha < 1.0f) {
    if (mp_alpha == 0.f) return static_cast<ScalarT>(0.f);
    MPTypeAccscalar mp_sample =
        static_cast<MPTypeAccscalar>(standard_uniform.sample());
    mp_scale *= COMPAT_POW(1 - mp_sample, 1.0f / mp_alpha);
    mp_alpha += 1.0f;
  }

  // This implements the acceptance-rejection method of Marsaglia and Tsang
  // (2000)
  // doi:10.1145/358407.358414
  const MPTypeAccscalar d = mp_alpha - 1.0f / 3.0f;
  const MPTypeAccscalar c = 1.0f / COMPAT_SQRT(9.0f * d);
  for (;;) {
    MPTypeAccscalar x, y;
    do {
      x = static_cast<MPTypeAccscalar>(standard_normal.sample());
      y = 1.0f + c * x;
    } while (y <= 0);
    const MPTypeAccscalar v = y * y * y;
    const MPTypeAccscalar u =
        1 - static_cast<MPTypeAccscalar>(standard_uniform.sample());
    const MPTypeAccscalar xx = x * x;
    if (u < 1.0f - 0.0331f * xx * xx)
      return static_cast<ScalarT>(mp_scale * d * v);
    if (COMPAT_LOG(u) < 0.5f * xx + d * (1.0f - v + COMPAT_LOG(v)))
      return static_cast<ScalarT>(mp_scale * d * v);
  }
}

template <typename T, typename UniformSamplerT, typename NormalSamplerT>
struct GammaCPUFunctor {
  GammaCPUFunctor(const T* alpha,
                  T* gamma,
                  BaseSampler<T, UniformSamplerT> uniform,
                  BaseSampler<T, NormalSamplerT> normal)
      : alpha_(alpha), gamma_(gamma), uniform_(uniform), normal_(normal) {}

  HOST void operator()(int64_t index) {
    auto sample = sample_gamma<T, T, UniformSamplerT, NormalSamplerT>(
        alpha_[index], uniform_, normal_);
    gamma_[index] = std::max(std::numeric_limits<T>::min(), sample);
  }

  const T* alpha_;
  T* gamma_;
  BaseSampler<T, UniformSamplerT> uniform_;
  BaseSampler<T, NormalSamplerT> normal_;
};

template <typename T>
struct GammaSampler<CPUContext, T> {
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out) {
    auto generator = dev_ctx.GetGenerator()->GetCPUEngine();

    auto uniform = [&generator]() -> T {
      std::uniform_real_distribution<T> u(0.0, 1.0);
      return u(*generator);
    };
    BaseSampler<T, decltype(uniform)> standard_uniform(uniform);

    auto normal = [&generator]() {
      std::normal_distribution<T> n(0.0, 1.0);
      return n(*generator);
    };
    BaseSampler<T, decltype(normal)> standard_normal(normal);

    GammaCPUFunctor<T, decltype(uniform), decltype(normal)> gamma_functor(
        alpha.data<T>(), out->data<T>(), standard_uniform, standard_normal);
    funcs::ForRange<CPUContext> for_range(dev_ctx, out->numel());
    for_range(gamma_functor);
  }
};

template <typename T>
struct DirichletSampler<CPUContext, T> {
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out) {
    // sample from K gamma distributions, where K=alpha.numel()
    DenseTensor gamma_samples;
    gamma_samples.Resize(alpha.dims());
    dev_ctx.template Alloc<T>(&gamma_samples);

    GammaSampler<CPUContext, T> gamma_sampler;
    gamma_sampler(dev_ctx, alpha, &gamma_samples);

    // normalize them into a simplex, along the last axis
    DenseTensor gamma_sum;
    auto new_shape = gamma_samples.dims();
    new_shape[new_shape.size() - 1] = 1;
    gamma_sum.Resize(new_shape);
    dev_ctx.template Alloc<T>(&gamma_sum);

    funcs::ReduceKernelImpl<CPUContext, T, T, funcs::SumFunctor>(
        dev_ctx,
        gamma_samples,
        &gamma_sum,
        {new_shape.size() - 1},
        true,
        false);

    funcs::ElementwiseCompute<funcs::DivideFunctor<T>, T>(
        dev_ctx, gamma_samples, gamma_sum, funcs::DivideFunctor<T>(), out);
  }
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
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
struct GammaSampler<GPUContext, T> {
  void operator()(const GPUContext& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out) {
    auto p_gen = dev_ctx.GetGenerator();
    auto seed_and_offset = p_gen->IncrementOffset(10);  // hard-coded offset
    auto seed = seed_and_offset.first;
    auto offset = seed_and_offset.second;

    GammaCUDAFunctor<T> gamma_functor(
        alpha.data<T>(), out->data<T>(), seed, offset);
    funcs::ForRange<GPUContext> for_range(dev_ctx, out->numel());
    for_range(gamma_functor);
  }
};

template <typename T>
struct DirichletSampler<GPUContext, T> {
  void operator()(const GPUContext& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out) {
    // sample from K gamma distributions, where K=alpha.numel()
    DenseTensor gamma_samples;
    gamma_samples.Resize(alpha.dims());
    dev_ctx.template Alloc<T>(&gamma_samples);

    GammaSampler<GPUContext, T> gamma_sampler;
    gamma_sampler(dev_ctx, alpha, &gamma_samples);

    // normalize them into a simplex, along the last axis
    DenseTensor gamma_sum;
    auto new_shape = gamma_samples.dims();
    new_shape[new_shape.size() - 1] = 1;
    gamma_sum.Resize(new_shape);
    dev_ctx.template Alloc<T>(&gamma_sum);

    phi::SumRawKernel<T, GPUContext>(dev_ctx,
                                     gamma_samples,
                                     {new_shape.size() - 1},
                                     true,
                                     false,
                                     gamma_sum.dtype(),
                                     &gamma_sum);
    phi::DivideKernel<T, GPUContext>(dev_ctx, gamma_samples, gamma_sum, out);
  }
};
#endif

template <typename T, typename Context>
void Dirichletkernel(const Context& dev_ctx,
                     const DenseTensor& alpha,
                     DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  DirichletSampler<Context, T> sampler;
  sampler(dev_ctx, alpha, out);
}

}  // namespace phi
