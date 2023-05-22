// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/polygamma_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename T>
__host__ __device__ T zeta(T x, T q) {
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
      -2.950130727918164224e12, /*1.067062284288e16/3617*/
      1.1646782814350067249e14, /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
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
  };

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
  __forceinline__ CudaPolygammaFunctor(int n) {
    _n = n;
  }
  __device__ __forceinline__ T operator()(const T _x) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(_x);
    const auto one = MT{1};
    return static_cast<T>(((_n % 2) ? one : -one) * std::exp(std::lgamma(static_cast<MT>(_n) + one)) * zeta<MT>(static_cast<MT>(_n + 1), mp_x));
  }
};

template <typename T, typename Context>
void PolygammaKernel(const Context& ctx, const DenseTensor& x, const int n, DenseTensor* out) {
  ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  auto functor = CudaPolygammaFunctor<T>(n);
  phi::funcs::ElementwiseKernel<T>(ctx, ins, &outs, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(polygamma, GPU, ALL_LAYOUT, phi::PolygammaKernel, float, double) {}
