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

#pragma once

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
static inline T ratevl(T x, const T num[], int64_t M,
    const T denom[], int64_t N) {
  // evaluating rational function, i.e., the ratio of two polynomials
  // the coefficients for numerator are given by `num` while coeffs for
  // denumerator are given by `denom`

  int64_t i, dir;
  T y, num_ans, denom_ans;
  T absx = ::fabs(x);
  const T *p;

  if (absx > 1) {
    /* Evaluate as a polynomial in 1/x. */
    dir = -1;
    p = num + M;
    y = 1 / x;
  }
  else {
    dir = 1;
    p = num;
    y = x;
  }

  /* Evaluate the numerator */
  num_ans = *p;
  p += dir;
  for (i = 1; i <= M; i++) {
    num_ans = num_ans * y + *p;
    p += dir;
  }
  /* Evaluate the denominator */
  if (absx > 1) {
    p = denom + N;
  }
  else {
    p = denom;
  }

  denom_ans = *p;
  p += dir;
  for (i = 1; i <= N; i++) {
    denom_ans = denom_ans * y + *p;
    p += dir;
  }
  if (absx > 1) {
    i = N - M;
    return ::pow(x, static_cast<T>(i)) * num_ans / denom_ans;
  }
  else {
    return num_ans / denom_ans;
  }
}

template <typename T>
static inline T lanczos_sum_expg_scaled(T x) {
  // lanczos approximation

  static const T lanczos_sum_expg_scaled_num[13] = {
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859
  };
  static const T lanczos_sum_expg_scaled_denom[13] = {
    1.,
    66.,
    1925.,
    32670.,
    357423.,
    2637558.,
    13339535.,
    45995730.,
    105258076.,
    150917976.,
    120543840.,
    39916800.,
    0
  };
  return ratevl(static_cast<T>(x), lanczos_sum_expg_scaled_num,
      sizeof(lanczos_sum_expg_scaled_num) / sizeof(lanczos_sum_expg_scaled_num[0]) - 1,
      lanczos_sum_expg_scaled_denom,
      sizeof(lanczos_sum_expg_scaled_denom) / sizeof(lanczos_sum_expg_scaled_denom[0]) - 1);
}

template <typename T>
static inline T _igam_helper_fac(T a, T x) {
  // compute x^a * exp(-a) / gamma(a)
  // corrected from (15) and (16) in [igam2] by replacing exp(x - a) with
  // exp(a - x).

  T ax, fac, res, num, numfac;
  static const T MAXLOG = std::is_same<T,double>::value ?
    7.09782712893383996843E2 : 88.72283905206835;
  static const T EXP1 = 2.718281828459045;
  static const T lanczos_g = 6.024680040776729583740234375;

  if (::fabs(a - x) > 0.4 * ::fabs(a)) {
    ax = a * ::log(x) - x - ::lgamma(a);
    if (ax < -MAXLOG) {
      return 0.0;
    }
    return ::exp(ax);
  }

  fac = a + lanczos_g - 0.5;
  res = ::sqrt(fac / EXP1) / lanczos_sum_expg_scaled(a);

  if ((a < 200) && (x < 200)) {
    res *= ::exp(a - x) * ::pow(x / fac, a);
  }
  else {
    num = x - a - lanczos_g + 0.5;
    numfac = num / fac;
    res *= ::exp(a * (::log1p(numfac) - numfac) + x * (0.5 - lanczos_g) / fac);
  }
  return res;
}

template <typename T>
static inline T _igam_helper_series(T a, T x) {
  // Compute igam using DLMF 8.11.4. [igam1]
  static const T MACHEP = std::is_same<T, double>::value ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;
  static const int MAXITER = 2000;

  int i;
  T ans, ax, c, r;

  ax = _igam_helper_fac(a, x);
  if (ax == 0.0) {
    return 0.0;
  }

  /* power series */
  r = a;
  c = 1.0;
  ans = 1.0;

  for (i = 0; i < MAXITER; i++) {
    r += 1.0;
    c *= x / r;
    ans += c;
    if (c <= MACHEP * ans) {
      break;
    }
  }
  return (ans * ax / a);
}

template <typename T>
struct IgammaFunctor {
    static const T SMALL = 20.0;
    static const T LARGE = 200.0;
    static const T SMALLRATIO = 0.3;
    static const T LARGERATIO = 4.5;
  IgammaFunctor(const T* input, const T* other, T* output, int64_t numel)
      : input_(input), other_(other), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
  /* the calculation of the regularized lower incomplete gamma function
   * is done differently based on the values of a and x:
   * - if x and/or a is at the boundary of defined region, then assign the
   *   result at the boundary
   * - if a is large and a ~ x, then using Uniform Asymptotic Expansions for
   *   Large Parameter (see DLMF 8.12.3 [igam1])
   * - if x > 1 and x > a, using the substraction from the regularized upper
   *   incomplete gamma
   * - otherwise, calculate the series from [igam2] eq (4)
   */
    a = other_[idx];
    x = input_[idx];
    
   // boundary values following SciPy
    if ((x < 0) || (a < 0)) {
    // out of defined-region of the function
    return std::numeric_limits<T>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 1.0;
    }
    else {
      return std::numeric_limits<T>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 0.0; // zero integration limit
  }
  else if (::isinf(static_cast<T>(a))) {
    if (::isinf(static_cast<T>(x))) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    return 0.0;
  }
  else if (::isinf(static_cast<T>(x))) {
    return 1.0;
  }

  /* Asymptotic regime where a ~ x. */
  absxma_a = ::fabs(x - a) / a;
  if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
    return _igam_helper_asymptotic_series(a, x, 1);
  }
  else if ((a > LARGE) && (absxma_a < LARGERATIO / ::sqrt(a))) {
    return _igam_helper_asymptotic_series(a, x, 1);
  }

  if ((x > 1.0) && (x > a)) {
    return 1.0 - calc_igammac(a, x);
  }

  return _igam_helper_series(a, x);
  }

 private:
  const T* input_;
  const T* other_;
  T* output_;
  int64_t numel_;
};

}  // namespace phi
