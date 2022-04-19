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

#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#endif

namespace phi {

#define DEFINE_CPU_ELEMENTWISE_OP(name)                                     \
  template <typename T, typename Context>                                   \
  void name##RawKernel(const Context& dev_ctx,                              \
                       const DenseTensor& x,                                \
                       const DenseTensor& y,                                \
                       int axis,                                            \
                       DenseTensor* out) {                                  \
    dev_ctx.template Alloc<T>(out);                                         \
    if (x.dims() == y.dims()) {                                             \
      SameDimsElementwiseCompute<SameDims##name##Functor<CPUContext, T>>()( \
          dev_ctx, x, y, out);                                              \
    } else {                                                                \
      auto x_dims = x.dims();                                               \
      auto y_dims = y.dims();                                               \
      if (x_dims.size() >= y_dims.size()) {                                 \
        funcs::ElementwiseCompute<funcs::name##Functor<T>, T>(              \
            dev_ctx, x, y, axis, funcs::name##Functor<T>(), out);           \
      } else {                                                              \
        funcs::ElementwiseCompute<funcs::Inverse##name##Functor<T>, T>(     \
            dev_ctx, x, y, axis, funcs::Inverse##name##Functor<T>(), out);  \
      }                                                                     \
    }                                                                       \
  }

#define DEFINE_CUDA_ELEMENTWISE_OP(name)                             \
  template <typename T, typename Context>                            \
  void name##RawKernel(const Context& dev_ctx,                       \
                       const DenseTensor& x,                         \
                       const DenseTensor& y,                         \
                       int axis,                                     \
                       DenseTensor* out) {                           \
    std::vector<const DenseTensor*> inputs;                          \
    std::vector<DenseTensor*> outputs;                               \
    inputs.emplace_back(&x);                                         \
    inputs.emplace_back(&y);                                         \
    outputs.emplace_back(out);                                       \
    dev_ctx.template Alloc<T>(out);                                  \
    funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(          \
        dev_ctx, inputs, &outputs, axis, funcs::name##Functor<T>()); \
  }

template <typename T, typename Context>
void FMaxKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                int axis,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::FMaxFunctor<T>, T, T>(
      dev_ctx, x, y, axis, funcs::FMaxFunctor<T>(), out);
}

template <typename T, typename Context>
void FMinKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                int axis,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::FMinFunctor<T>, T, T>(
      dev_ctx, x, y, axis, funcs::FMinFunctor<T>(), out);
}

}  // namespace phi
