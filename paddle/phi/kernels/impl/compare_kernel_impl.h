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
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"

namespace phi {

template <typename T,
          typename Context,
          typename Functor,
          typename InverseFunctor>
inline void CompareKernelImpl(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              int axis,
                              DenseTensor* out);

template <typename T,
          typename Context,
          typename Functor,
          typename InverseFunctor>
inline void InplaceCompareKernelImpl(const Context& dev_ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& y,
                                     int axis,
                                     DenseTensor* out);

template <typename T, typename Context, typename Functor>
inline void CompareAllKernelImpl(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 DenseTensor* out);

#define DEFINE_COMPARE_KERNEL(name, functor, inverse_functor)               \
  template <typename T, typename Context>                                   \
  void name##Kernel(const Context& dev_ctx,                                 \
                    const DenseTensor& x,                                   \
                    const DenseTensor& y,                                   \
                    DenseTensor* out) {                                     \
    if (out->IsSharedWith(x)) {                                             \
      InplaceCompareKernelImpl<T, Context, functor<T>, inverse_functor<T>>( \
          dev_ctx, x, y, -1, out);                                          \
    } else {                                                                \
      CompareKernelImpl<T, Context, functor<T>, inverse_functor<T>>(        \
          dev_ctx, x, y, -1, out);                                          \
    }                                                                       \
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

#define DEFINE_COMPARE_ALL_KERNEL(compare_all_kernel, functor)          \
  template <typename T, typename Context>                               \
  void compare_all_kernel(const Context& dev_ctx,                       \
                          const DenseTensor& x,                         \
                          const DenseTensor& y,                         \
                          DenseTensor* out) {                           \
    if (x.dtype() == y.dtype()) {                                       \
      CompareAllKernelImpl<T, Context, funcs::EqualFunctor<T>>(         \
          dev_ctx, x, y, out);                                          \
      return;                                                           \
    }                                                                   \
    DenseTensor x_dbl, y_dbl;                                           \
    x_dbl.Resize(x.dims());                                             \
    y_dbl.Resize(y.dims());                                             \
    dev_ctx.template Alloc<double>(&x_dbl);                             \
    dev_ctx.template Alloc<double>(&y_dbl);                             \
    PD_VISIT_ALL_TYPES(x.dtype(), "EqualAllKernel_CastX", ([&] {        \
                         x_dbl = phi::Cast<data_t, Context>(            \
                             dev_ctx, x, phi::DataType::FLOAT64);       \
                       }));                                             \
    y_dbl = phi::Cast<T, Context>(dev_ctx, y, phi::DataType::FLOAT64);  \
    CompareAllKernelImpl<double, Context, funcs::EqualFunctor<double>>( \
        dev_ctx, x_dbl, y_dbl, out);                                    \
  }

DEFINE_COMPARE_ALL_KERNEL(EqualAllKernel, funcs::EqualFunctor)
#undef DEFINE_COMPARE_ALL_KERNEL

}  // namespace phi
