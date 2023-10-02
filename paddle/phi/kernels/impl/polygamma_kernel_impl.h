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
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#else
#include "paddle/phi/kernels/funcs/for_range.h"
#endif

namespace phi {

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
__host__ __device__ T zeta(T x, T q) {
  /*
   * REFERENCE:
   * Gradshteyn, I. S., and I. M. Ryzhik, Tables of Integrals,
   * Series, and Products, p. 1073; Academic Press, 1980.
   * From https://netlib.org/cephes/doubldoc.html - zeta.c
   */
  const T MACHEP = T{1.11022302462515654042E-16};
  constexpr T zero = T{0.0};
  constexpr T half = T{0.5};
  constexpr T one = T{1.0};
  static const T A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12,  /*1.067062284288e16/3617*/
      1.1646782814350067249e14,  /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17,  /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091*/
  };

  int i = 0;
  T a, b, k, s, t, w;
  if (x == one) {
    return std::numeric_limits<T>::infinity();
  }

  if (x < one) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (q <= zero) {
    if (q == std::floor(q)) {
      return std::numeric_limits<T>::infinity();
    }
    if (x != std::floor(x)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
  }

  s = std::pow(q, -x);
  a = q;
  i = 0;
  b = zero;
  while ((i < 9) || (a <= T{9.0})) {
    i += 1;
    a += one;
    b = ::pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return static_cast<T>(s);
    }
  }

  w = a;
  s += b * w / (x - one);
  s -= half * b;
  a = one;
  k = zero;
  for (int i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = std::fabs(t / s);
    if (t < MACHEP) {
      return static_cast<T>(s);
    }
    k += one;
    a *= x + k;
    b /= w;
    k += one;
  }
  return static_cast<T>(s);
}

template <typename T>
struct CudaPolygammaFunctor {
  int _n;
  __forceinline__ CudaPolygammaFunctor(int n) { _n = n; }
  __device__ __forceinline__ T operator()(const T _x) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(_x);
    const auto one = MT{1};
    return static_cast<T>(((_n % 2) ? one : -one) *
                          std::exp(std::lgamma(static_cast<MT>(_n) + one)) *
                          zeta<MT>(static_cast<MT>(_n + 1), mp_x));
  }
};

template <typename T>
struct CudaPolygammaGradFunctor {
  int _n;
  __forceinline__ CudaPolygammaGradFunctor(int n) { _n = n; }
  __device__ __forceinline__ T operator()(const T _x, const T _out_grad) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(_x);
    const MT mp_out_grad = static_cast<MT>(_out_grad);
    const auto one = MT{1};
    return static_cast<T>(mp_out_grad * ((_n % 2) ? one : -one) *
                          std::exp(std::lgamma(static_cast<MT>(_n) + one)) *
                          zeta<MT>(static_cast<MT>(_n + 1), mp_x));
  }
};
#else
template <typename T>
static inline T zeta(T x, T q) {
  /*
   * REFERENCE:
   * Gradshteyn, I. S., and I. M. Ryzhik, Tables of Integrals,
   * Series, and Products, p. 1073; Academic Press, 1980.
   * From https://netlib.org/cephes/doubldoc.html - zeta.c
   */
  const T MACHEP = T{1.11022302462515654042E-16};
  constexpr T zero = T{0.0};
  constexpr T half = T{0.5};
  constexpr T one = T{1.0};
  static const T A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12,  /*1.067062284288e16/3617*/
      1.1646782814350067249e14,  /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17,  /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091*/
  };

  int i = 0;
  T a, b, k, s, t, w;
  if (x == one) {
    return std::numeric_limits<T>::infinity();
  }

  if (x < one) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (q <= zero) {
    if (q == std::floor(q)) {
      return std::numeric_limits<T>::infinity();
    }
    if (x != std::floor(x)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
  }

  s = std::pow(q, -x);
  a = q;
  i = 0;
  b = zero;
  while ((i < 9) || (a <= T{9.0})) {
    i += 1;
    a += one;
    b = std::pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return static_cast<T>(s);
    }
  }

  w = a;
  s += b * w / (x - one);
  s -= half * b;
  a = one;
  k = zero;
  for (int i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = std::fabs(t / s);
    if (t < MACHEP) {
      return static_cast<T>(s);
    }
    k += one;
    a *= x + k;
    b /= w;
    k += one;
  }
  return static_cast<T>(s);
}

template <typename T>
struct PolygammaFunctor {
  PolygammaFunctor(const T* input, const int n, T* output, int64_t size)
      : input_(input), n_(n), output_(output), size_(size) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(input_[idx]);

    const auto one = MT{1};
    output_[idx] =
        static_cast<T>(((n_ % 2) ? one : -one) *
                       std::exp(std::lgamma(static_cast<MT>(n_) + one)) *
                       zeta<MT>(static_cast<MT>(n_ + 1), mp_x));
  }

 private:
  const T* input_;
  const int n_;
  T* output_;
  int64_t size_;
};

template <typename T>
struct PolygammaGradFunctor {
  PolygammaGradFunctor(
      const T* input, const int n, const T* out_grad, T* output, int64_t size)
      : input_(input),
        n_(n),
        out_grad_(out_grad),
        output_(output),
        size_(size) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(input_[idx]);
    const MT mp_out_grad = static_cast<MT>(out_grad_[idx]);

    const auto one = MT{1};
    auto partial_x = ((n_ % 2) ? one : -one) *
                     std::exp(std::lgamma(static_cast<MT>(n_) + one)) *
                     zeta<MT>(static_cast<MT>(n_ + 1), mp_x);
    output_[idx] = static_cast<T>(mp_out_grad * partial_x);
  }

 private:
  const T* input_;
  const int n_;
  const T* out_grad_;
  T* output_;
  int64_t size_;
};
#endif

}  // namespace phi
