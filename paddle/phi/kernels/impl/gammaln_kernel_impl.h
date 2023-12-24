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
struct GammalnFunctor {
  GammalnFunctor(const T* x, T* output, int64_t numel)
      : x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(x_[idx]);
    output_[idx] = static_cast<T>(std::lgamma(mp_x));
  }

 private:
  const T* x_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Context>
void GammalnKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out) {
  auto numel = x.numel();
  auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);
  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  GammalnFunctor<T> functor(x_data, out_data, numel);
  for_range(functor);
}
}  // namespace phi
