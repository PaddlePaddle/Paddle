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
#include <cmath>
#include <random>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

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

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
struct DirichletSampler;

template <typename ScalarT, typename SamplerT>
struct BaseSampler {
  SamplerT sampler_;
  HOSTDEVICE BaseSampler(const SamplerT& sampler) : sampler_(sampler) {}
  HOSTDEVICE ScalarT sample() { return sampler_(); }
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

template <typename ScalarT, typename AccscalarT, typename UniformSamplerT,
          typename NormalSamplerT>
HOSTDEVICE ScalarT sample_gamma(
    ScalarT alpha, BaseSampler<AccscalarT, UniformSamplerT> standard_uniform,
    BaseSampler<AccscalarT, NormalSamplerT> standard_normal) {
  AccscalarT scale = 1.0f;

  // Boost alpha for higher acceptance probability.
  if (alpha < 1.0f) {
    if (alpha == 0.f) return 0.f;
    scale *= COMPAT_POW(1 - standard_uniform.sample(), 1.0f / alpha);
    alpha += 1.0f;
  }

  // This implements the acceptance-rejection method of Marsaglia and Tsang
  // (2000)
  // doi:10.1145/358407.358414
  const AccscalarT d = alpha - 1.0f / 3.0f;
  const AccscalarT c = 1.0f / COMPAT_SQRT(9.0f * d);
  for (;;) {
    AccscalarT x, y;
    do {
      x = standard_normal.sample();
      y = 1.0f + c * x;
    } while (y <= 0);
    const AccscalarT v = y * y * y;
    const AccscalarT u = 1 - standard_uniform.sample();
    const AccscalarT xx = x * x;
    if (u < 1.0f - 0.0331f * xx * xx)
      return static_cast<ScalarT>(scale * d * v);
    if (COMPAT_LOG(u) < 0.5f * xx + d * (1.0f - v + COMPAT_LOG(v)))
      return static_cast<ScalarT>(scale * d * v);
  }
}

template <typename DeviceContext, typename T>
class DirichletKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* alpha = ctx.Input<framework::Tensor>("Alpha");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    DirichletSampler<DeviceContext, T> sampler;
    sampler(ctx, alpha, out);
  }
};
}  // namespace operators
}  // namespace paddle
