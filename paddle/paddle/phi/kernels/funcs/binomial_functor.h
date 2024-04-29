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

#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {

template <typename T>
inline T stirling_approx_tail(int64_t k) {
  const T kTailValues[] = {0.0810614667953272,
                           0.0413406959554092,
                           0.0276779256849983,
                           0.02079067210376509,
                           0.0166446911898211,
                           0.0138761288230707,
                           0.0118967099458917,
                           0.0104112652619720,
                           0.00925546218271273,
                           0.00833056343336287};
  if (k <= 9) {
    return static_cast<T>(kTailValues[static_cast<size_t>(k)]);
  }
  T kp1sq = (k + 1) * (k + 1);
  return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
}

template <typename T, typename Context>
inline int64_t btrs(const Context& ctx, const T n, const T p) {
  int64_t k;
  T U, V, us;
  std::uniform_real_distribution<T> dist(0.0, 1.0);
  auto gen_ptr = ctx.GetGenerator();
  auto engine = gen_ptr->GetCPUEngine();

  const T stddev = std::sqrt(n * p * (1 - p));

  const T b = 1.15 + 2.53 * stddev;
  const T a = -0.0873 + 0.0248 * b + 0.01 * p;
  const T c = n * p + 0.5;
  const T v_r = 0.92 - 4.2 / b;
  const T r = p / (1 - p);

  const T alpha = (2.83 + 5.1 / b) * stddev;
  const T m = std::floor((n + 1) * p);

  while (1) {
    U = dist(*engine) - 0.5;
    V = dist(*engine);

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

template <typename T, typename Context>
inline int64_t binomial_inversion(const Context& ctx, const T n, const T p) {
  T unif;
  T geom_sum = 0.0;
  int64_t num_geom = 0;
  T logprob = std::log1p(-p);
  std::uniform_real_distribution<T> dist(0.0, 1.0);
  auto gen_ptr = ctx.GetGenerator();
  auto engine = gen_ptr->GetCPUEngine();

  while (1) {
    unif = dist(*engine);
    T geom = std::ceil(std::log(unif) / logprob);
    geom_sum += geom;
    if (geom_sum > n) {
      break;
    }
    num_geom = num_geom + 1;
  }
  return num_geom;
}

template <typename T, typename Context>
inline int64_t BinomialFunctor(const Context& ctx, const T n, const T p) {
  if (n <= 0.0 || p <= 0.0) {
    return 0;
  } else if (p >= 1.0) {
    return static_cast<int64_t>(n);
  } else if (p <= 0.5) {
    if (n * p >= 10.0) {
      return btrs<T>(ctx, n, p);
    } else {
      return binomial_inversion<T>(ctx, n, p);
    }
  } else {
    T qprob = 1.0 - p;
    if (n * qprob >= 10.0) {
      return static_cast<int64_t>(n) - btrs<T>(ctx, n, qprob);
    } else {
      return static_cast<int64_t>(n) - binomial_inversion<T>(ctx, n, qprob);
    }
  }
}

}  // namespace funcs
}  // namespace phi
