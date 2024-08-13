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

#pragma once

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/for_range.h"

#define MAXLOG 7.09782712893383996732E2
#define MACHEP 1.11022302462515654042E-16

namespace phi {
template <typename T>
HOSTDEVICE T igam(const T a, const T x);
template <typename T>
HOSTDEVICE T igamc(const T a, const T x);

template <typename T>
HOSTDEVICE T igam(const T a, const T x) {
  if ((x <= T{0}) || (a <= T{0})) return (T{0.0});

  if ((x > T{1.0}) && (x > a)) return (T{1.0} - igamc(a, x));

  /* Compute  x**a * exp(-x) / gamma(a)  */
  T ax = a * log(x) - x - std::lgamma(a);
  if (ax < -MAXLOG) {
    return (T{0.0});
  }
  ax = exp(ax);

  /* power series */
  T r = a;
  T c = T{1.0};
  T ans = T{1.0};

  do {
    r += T{1.0};
    c *= x / r;
    ans += c;
  } while (c / ans > MACHEP);

  return (ans * ax / a);
}

template <typename T>
HOSTDEVICE T igamc(const T a, const T x) {
  static T big = 4.503599627370496e15;
  static T biginv = 2.22044604925031308085e-16;

  if ((x <= T{0}) || (a <= T{0})) return (T{1.0});

  if ((x < T{1.0}) || (x < a)) return (T{1.0} - igam(a, x));

  T ax = a * log(x) - x - std::lgamma(a);
  if (ax < -MAXLOG) {
    return (T{0.0});
  }
  ax = exp(ax);

  /* continued fraction */
  T y = T{1.0} - a;
  T z = x + y + T{1.0};
  T c = T{0.0};
  T pkm2 = T{1.0};
  T qkm2 = x;
  T pkm1 = x + T{1.0};
  T qkm1 = z * x;
  T ans = pkm1 / qkm1;
  T t;
  do {
    c += T{1.0};
    y += T{1.0};
    z += T{2.0};
    T yc = y * c;
    T pk = pkm1 * z - pkm2 * yc;
    T qk = qkm1 * z - qkm2 * yc;
    if (qk != T{0}) {
      T r = pk / qk;
      t = fabs((ans - r) / r);
      ans = r;
    } else {
      t = T{1.0};
    }
    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;
    if (fabs(pk) > big) {
      pkm2 *= biginv;
      pkm1 *= biginv;
      qkm2 *= biginv;
      qkm1 *= biginv;
    }
  } while (t > MACHEP);

  return (ans * ax);
}

template <typename T>
struct IgammaFunctor {
  IgammaFunctor(const T* x, const T* a, T* output, int64_t numel)
      : x_(x), a_(a), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(x_[idx]);
    const MT mp_a = static_cast<MT>(a_[idx]);
    output_[idx] = static_cast<T>(igamc<MT>(mp_a, mp_x));
  }

 private:
  const T* x_;
  const T* a_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Context>
void GammainccKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  auto numel = x.numel();
  auto* x_data = x.data<T>();
  auto* y_data = y.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);
  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  IgammaFunctor<T> functor(y_data, x_data, out_data, numel);
  for_range(functor);
}
}  // namespace phi
