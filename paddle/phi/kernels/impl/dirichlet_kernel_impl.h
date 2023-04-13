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
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/dirichlet_kernel.h"

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

template <typename Context, typename T>
struct DirichletSampler {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& alpha,
                  DenseTensor* out);
};

template <typename T, typename Context>
void Dirichletkernel(const Context& dev_ctx,
                     const DenseTensor& alpha,
                     DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  DirichletSampler<Context, T> sampler;
  sampler(dev_ctx, alpha, out);
}
}  // namespace phi
