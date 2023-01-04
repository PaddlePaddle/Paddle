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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"

namespace phi {

template <typename T,
          typename Context,
          typename Functor,
          typename InverseFunctor>
inline void CompareKernelImpl(const Context& ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              int axis,
                              DenseTensor* out);

template <typename T, typename Context, typename Functor>
inline void CompareAllKernelImpl(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 DenseTensor* out);

#define DEFINE_COMPARE_KERNEL(name, functor, inverse_functor)      \
  template <typename T, typename Context>                          \
  void name##RawKernel(const Context& ctx,                         \
                       const DenseTensor& x,                       \
                       const DenseTensor& y,                       \
                       int axis,                                   \
                       DenseTensor* out) {                         \
    CompareKernelImpl<T, Context, functor<T>, inverse_functor<T>>( \
        ctx, x, y, axis, out);                                     \
  }                                                                \
  template <typename T, typename Context>                          \
  void name##Kernel(const Context& ctx,                            \
                    const DenseTensor& x,                          \
                    const DenseTensor& y,                          \
                    DenseTensor* out) {                            \
    name##RawKernel<T, Context>(ctx, x, y, -1, out);               \
  }

DEFINE_COMPARE_KERNEL(LessThan,
                      funcs::LessThanFunctor,
                      funcs::GreaterThanFunctor)
DEFINE_COMPARE_KERNEL(LessEqual,
                      funcs::LessEqualFunctor,
                      funcs::GreaterEqualFunctor)
DEFINE_COMPARE_KERNEL(GreaterThan,
                      funcs::GreaterThanFunctor,
                      funcs::LessThanFunctor)
DEFINE_COMPARE_KERNEL(GreaterEqual,
                      funcs::GreaterEqualFunctor,
                      funcs::LessEqualFunctor)
DEFINE_COMPARE_KERNEL(Equal, funcs::EqualFunctor, funcs::EqualFunctor)
DEFINE_COMPARE_KERNEL(NotEqual, funcs::NotEqualFunctor, funcs::NotEqualFunctor)
#undef DEFINE_COMPARE_KERNEL

#define DEFINE_COMPARE_ALL_KERNEL(compare_all_kernel, functor)    \
  template <typename T, typename Context>                         \
  void compare_all_kernel(const Context& ctx,                     \
                          const DenseTensor& x,                   \
                          const DenseTensor& y,                   \
                          DenseTensor* out) {                     \
    CompareAllKernelImpl<T, Context, functor<T>>(ctx, x, y, out); \
  }

DEFINE_COMPARE_ALL_KERNEL(EqualAllKernel, funcs::EqualFunctor)
#undef DEFINE_COMPARE_ALL_KERNEL

}  // namespace phi
