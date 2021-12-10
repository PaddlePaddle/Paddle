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

#pragma once
#include <iostream>
#include <random>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

// ROCM hcc doesn't work well with using std:: in kernel functions
#if defined(__CUDA_ARCH__)
#else
#define compat_exp std::exp
#define compat_ceil std::ceil
#define compat_floor std::floor
#define compat_log std::log
#define compat_pow std::pow
#define compat_sqrt std::sqrt
#define compat_tan std::tan
#define compat_abs std::abs
#define compat_log1p std::log1p
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename scalar_t, typename sampler_t>
struct BaseSampler {
  sampler_t sampler_;
  HOSTDEVICE BaseSampler(const sampler_t& sampler) : sampler_(sampler) {}
  HOSTDEVICE scalar_t sample() { return sampler_(); }
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

template <typename scalar_t, typename accscalar_t, typename uniform_sampler_t,
          typename normal_sampler_t>
HOSTDEVICE scalar_t
sample_gamma(scalar_t alpha,
             BaseSampler<accscalar_t, uniform_sampler_t> standard_uniform,
             BaseSampler<accscalar_t, normal_sampler_t> standard_normal) {
  accscalar_t scale = 1.0f;

  // Boost alpha for higher acceptance probability.
  if (alpha < 1.0f) {
    if (alpha == 0.f) return 0.f;
    scale *= compat_pow(1 - standard_uniform.sample(), 1.0f / alpha);
    alpha += 1.0f;
  }

  // This implements the acceptance-rejection method of Marsaglia and Tsang
  // (2000)
  // doi:10.1145/358407.358414
  const accscalar_t d = alpha - 1.0f / 3.0f;
  const accscalar_t c = 1.0f / compat_sqrt(9.0f * d);
  for (;;) {
    accscalar_t x, y;
    do {
      x = standard_normal.sample();
      y = 1.0f + c * x;
    } while (y <= 0);
    const accscalar_t v = y * y * y;
    const accscalar_t u = 1 - standard_uniform.sample();
    const accscalar_t xx = x * x;
    if (u < 1.0f - 0.0331f * xx * xx)
      return static_cast<scalar_t>(scale * d * v);
    if (compat_log(u) < 0.5f * xx + d * (1.0f - v + compat_log(v)))
      return static_cast<scalar_t>(scale * d * v);
  }
}

template <typename T, typename uniform_sampler_t, typename normal_sampler_t>
struct GammaSampler {
  GammaSampler(const T* alpha, T* gamma,
               BaseSampler<T, uniform_sampler_t> uniform,
               BaseSampler<T, normal_sampler_t> normal)
      : alpha_(alpha), gamma_(gamma), uniform_(uniform), normal_(normal) {}

  HOSTDEVICE void operator()(int64_t index) {
    auto sample = sample_gamma<T, T, uniform_sampler_t, normal_sampler_t>(
        alpha_[index], uniform_, normal_);
    gamma_[index] = std::max(std::numeric_limits<T>::min(), sample);
  }

  const T* alpha_;
  T* gamma_;
  BaseSampler<T, uniform_sampler_t> uniform_;
  BaseSampler<T, normal_sampler_t> normal_;
};

template <typename DeviceContext, typename T>
struct DirichletSampler {
  void operator()(const framework::ExecutionContext& ctx, const Tensor* alpha,
                  Tensor* out);
};

template <typename DeviceContext, typename T>
class DirichletKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* alpha = ctx.Input<Tensor>("Alpha");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    DirichletSampler<DeviceContext, T> sample;
    sample(ctx, alpha, out);
  }
};

}  // namespace operators
}  // namespace paddle
