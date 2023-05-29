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

#include "paddle/phi/kernels/funcs/isfinite_functor.h"
#include "paddle/phi/kernels/isfinite_kernel.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/common/transform.h"

namespace phi {

#define DEFINE_ISFINITE_KERNEL(isfinite_kernel, functor)                   \
  template <typename T, typename Context>                                  \
  void isfinite_kernel(                                                    \
      const Context& ctx, const DenseTensor& x, DenseTensor* out) {        \
    auto* out_ptr = ctx.template Alloc<bool>(out);                         \
    funcs::functor<T> unary_func;                                          \
    phi::Transform<Context> trans;                                         \
    trans(ctx, x.data<T>(), x.data<T>() + x.numel(), out_ptr, unary_func); \
  }

DEFINE_ISFINITE_KERNEL(IsinfKernel, IsInfFunctor)
DEFINE_ISFINITE_KERNEL(IsnanKernel, IsNanFunctor)
DEFINE_ISFINITE_KERNEL(IsfiniteKernel, IsFiniteFunctor)
#undef DEFINE_ISFINITE_KERNEL

}  // namespace phi
