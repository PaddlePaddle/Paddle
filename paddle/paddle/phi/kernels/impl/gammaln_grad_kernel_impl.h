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
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {
template <typename T>
HOSTDEVICE T digamma(T x) {
  static T c = T{8.5};
  static T euler_mascheroni = T{0.57721566490153286060};
  T r;
  T value;
  T x2;

  if (x <= T{0.0}) {
    value = T{0.0};
    return value;
  }

  if (x <= T{0.000001}) {
    value = -euler_mascheroni - T{1.0} / x + T{1.6449340668482264365} * x;
    return value;
  }

  value = T{0.0};
  x2 = x;
  while (x2 < c) {
    value = value - T{1.0} / x2;
    x2 = x2 + T{1.0};
  }

  r = T{1.0} / x2;
  value = value + std::log(x2) - T{0.5} * r;

  r = r * r;

  value = value -
          r * (T{1.0} / T{12.0} -
               r * (T{1.0} / T{120.0} -
                    r * (T{1.0} / T{252.0} -
                         r * (T{1.0} / T{240.0} - r * (T{1.0} / T{132.0})))));

  return value;
}

template <typename T>
struct GammalnGradFunctor {
  GammalnGradFunctor(const T* dout, const T* x, T* output, int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_dout = static_cast<MT>(dout_[idx]);
    const MT mp_x = static_cast<MT>(x_[idx]);
    output_[idx] = static_cast<T>(mp_dout * digamma<MT>(mp_x));
  }

 private:
  const T* dout_;
  const T* x_;
  T* output_;
  int64_t numel_;
};
template <typename T, typename Context>
void GammalnGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& d_out,
                       DenseTensor* d_x) {
  auto numel = d_out.numel();
  auto* dout_data = d_out.data<T>();
  auto* x_data = x.data<T>();
  auto* dx_data =
      dev_ctx.template Alloc<T>(d_x, static_cast<size_t>(numel * sizeof(T)));
  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  GammalnGradFunctor<T> functor(dout_data, x_data, dx_data, numel);
  for_range(functor);
}
}  // namespace phi
