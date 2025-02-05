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

namespace phi {
template <typename T>
struct IgammaGradFunctor {
  IgammaGradFunctor(
      const T* dout, const T* x, const T* a, T* output, int64_t numel)
      : dout_(dout), x_(x), a_(a), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_dout = static_cast<MT>(dout_[idx]);
    const MT mp_x = static_cast<MT>(x_[idx]);
    const MT mp_a = static_cast<MT>(a_[idx]);
    const MT mp_a_1 = static_cast<MT>(a_[idx] - 1);
    output_[idx] = static_cast<T>(mp_dout * -std::exp(-mp_x) *
                                  std::pow(mp_x, mp_a_1) / std::tgamma(mp_a));
  }

 private:
  const T* dout_;
  const T* x_;
  const T* a_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Context>
void GammainccGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& d_out,
                         DenseTensor* d_y) {
  auto numel = d_out.numel();
  auto* dout_data = d_out.data<T>();
  auto* x_data = x.data<T>();
  auto* y_data = y.data<T>();
  auto* dy_data =
      dev_ctx.template Alloc<T>(d_y, static_cast<size_t>(numel * sizeof(T)));
  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  IgammaGradFunctor<T> functor(dout_data, y_data, x_data, dy_data, numel);
  for_range(functor);
}
}  // namespace phi
