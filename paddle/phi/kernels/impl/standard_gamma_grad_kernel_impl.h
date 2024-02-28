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

#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/standard_gamma_grad_kernel.h"

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

template <typename T>
HOSTDEVICE T polevl(const T x, const T A[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

template <typename ScalarT>
HOSTDEVICE ScalarT digamma(ScalarT x_) {
  using MPTypeScalar = typename phi::dtype::MPTypeTrait<ScalarT>::Type;

  MPTypeScalar x = static_cast<MPTypeScalar>(x_);
  constexpr MPTypeScalar PSI_10 = 2.25175258906672110764;
  constexpr MPTypeScalar PI = 3.14159265358979323846;

  if (x == 0) {
    return INFINITY;
  }
  MPTypeScalar additional_summand = 0;
  int x_is_integer = x == COMPAT_FLOOR(x);
  if (x < 0) {
    if (x_is_integer) {
      return INFINITY;
    }
    // it is more standard to write this as recursion, but
    // nvcc does not like that
    additional_summand = -PI / COMPAT_TAN(PI * x);
    x = 1 - x;
  }

  // Push x to be >= 10
  MPTypeScalar result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return result + PSI_10 + additional_summand;
  }

  // Compute asymptotic digamma
  static const MPTypeScalar A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  MPTypeScalar y = 0;
  if (x < 1.0e17f) {
    MPTypeScalar z = 1.0 / (x * x);
    y = z * polevl<MPTypeScalar>(z, A, 6);
  }
  return static_cast<ScalarT>(result + COMPAT_LOG(x) - (0.5f / x) - y +
                              additional_summand);
}

// Computes the reparameterized gradient -(d/dalpha cdf(x;alpha)) / pdf(x;alpha)
// for random number x drawn from a standard Gamma distribution Gamma(alpha).
template <typename ScalarT>
HOSTDEVICE ScalarT gamma_grad(ScalarT alpha_, ScalarT x_) {
  using MPTypeScalar = typename phi::dtype::MPTypeTrait<ScalarT>::Type;
  MPTypeScalar alpha = static_cast<MPTypeScalar>(alpha_);
  MPTypeScalar x = static_cast<MPTypeScalar>(x_);

  if (x < 0.8f) {
    MPTypeScalar numer = 1.0f;
    MPTypeScalar denom = alpha;
    auto series1 = numer / denom;
    auto series2 = numer / (denom * denom);
    for (int i = 1; i <= 5; ++i) {
      numer *= -x / static_cast<MPTypeScalar>(i);
      denom += 1;
      series1 += numer / denom;
      series2 += numer / (denom * denom);
    }
    const auto pow_x_alpha = COMPAT_POW(x, alpha);
    const auto gamma_pdf = COMPAT_POW(x, alpha - 1) * COMPAT_EXP(-x);
    const auto gamma_cdf = pow_x_alpha * series1;
    const auto gamma_cdf_alpha =
        (COMPAT_LOG(x) - digamma<MPTypeScalar>(alpha)) * gamma_cdf -
        pow_x_alpha * series2;
    const auto result = -gamma_cdf_alpha / gamma_pdf;
    return std::isnan(result) ? static_cast<ScalarT>(0.f)
                              : static_cast<ScalarT>(result);
  }

  // Use a Rice saddle point expansion for large alpha.
  if (alpha > 8.0f) {
    if (0.9f * alpha <= x && x <= 1.1f * alpha) {
      const auto numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
      const auto numer_2 = 1440 * (alpha * alpha) + 6 * x * (53 - 120 * x) -
                           65 * x * x / alpha + alpha * (107 + 3600 * x);
      const auto denom = 1244160 * (alpha * alpha) * (alpha * alpha);
      return static_cast<ScalarT>(numer_1 * numer_2 / denom);
    }
    const auto denom = COMPAT_SQRT(8 * alpha);
    const auto term2 = denom / (alpha - x);
    const auto term3 = COMPAT_POW(x - alpha - alpha * COMPAT_LOG(x / alpha),
                                  static_cast<MPTypeScalar>(-1.5));
    const auto term23 = (x < alpha) ? term2 - term3 : term2 + term3;
    const auto term1 =
        COMPAT_LOG(x / alpha) * term23 -
        COMPAT_SQRT(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
    const auto stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
    const auto numer = x * term1;
    return static_cast<ScalarT>(-stirling * numer / denom);
  }

  // Use a bivariate rational approximation to the reparameterized gradient.
  const auto u = COMPAT_LOG(x / alpha);
  const auto v = COMPAT_LOG(alpha);
  static const MPTypeScalar coef_uv[3][8] = {
      {0.16009398,
       -0.094634809,
       0.025146376,
       -0.0030648343,
       1,
       0.32668115,
       0.10406089,
       0.0014179084},
      {0.53487893,
       0.1298071,
       0.065735949,
       -0.0015649758,
       0.16639465,
       0.020070113,
       -0.0035938915,
       -0.00058392623},
      {0.040121004,
       -0.0065914022,
       -0.0026286047,
       -0.0013441777,
       0.017050642,
       -0.0021309326,
       0.00085092367,
       -1.5247877e-07},
  };
  MPTypeScalar coef_v[8];
  for (int i = 0; i < 8; ++i) {
    coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
  }
  const auto p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  const auto q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  return static_cast<ScalarT>(COMPAT_EXP(p / q));
}

template <typename T>
struct GammaGradFunctor {
  GammaGradFunctor(const T* alpha, const T* x, T* grad)
      : alpha_(alpha), x_(x), grad_(grad) {}

  HOSTDEVICE void operator()(int64_t index) {
    grad_[index] = gamma_grad<T>(alpha_[index], x_[index]);
    // std::max(std::numeric_limits<T>::min(), sample);
  }

  const T* alpha_;
  const T* x_;
  T* grad_;
};

template <typename T, typename Context>
void StandardGammaGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& out,
                             DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  GammaGradFunctor<T> gamma_grad_functor(
      x.data<T>(), out.data<T>(), x_grad->data<T>());
  funcs::ForRange<Context> for_range(dev_ctx, x_grad->numel());
  for_range(gamma_grad_functor);
}

}  // namespace phi
