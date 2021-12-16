/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/backends/cuda/cuda_context.h"
#include "paddle/pten/core/dense_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"

namespace pten {
template <typename T, typename ContextT>
void Add(const ContextT& dev_ctx,
         const DenseTensor& x,
         const DenseTensor& y,
         int axis,
         DenseTensor* out);

template <typename T, typename ContextT>
void Subtract(const ContextT& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& y,
              int axis,
              DenseTensor* out);

template <typename T, typename ContextT>
void Divide(const ContextT& dev_ctx,
            const DenseTensor& x,
            const DenseTensor& y,
            int axis,
            DenseTensor* out);

template <typename T, typename ContextT>
void Multiply(const ContextT& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& y,
              int axis,
              DenseTensor* out);

}  // namespace pten

#define DEFINE_CPU_ELEMENTWISE_OP(name)                                        \
  template <typename T, typename ContextT>                                     \
  void name(const ContextT& dev_ctx,                                           \
            const DenseTensor& x,                                              \
            const DenseTensor& y,                                              \
            int axis,                                                          \
            DenseTensor* out) {                                                \
    out->mutable_data<T>();                                                    \
    if (x.dims() == y.dims()) {                                                \
      SameDimsElementwiseCompute<SameDims##name##Functor<ContextT, T>>()(      \
          dev_ctx, x, y, out);                                                 \
    } else {                                                                   \
      auto x_dims = x.dims();                                                  \
      auto y_dims = y.dims();                                                  \
      if (x_dims.size() >= y_dims.size()) {                                    \
        ElementwiseCompute<functions::name##Functor<T>, T>(                    \
            dev_ctx, x, y, axis, functions::name##Functor<T>(), out);          \
      } else {                                                                 \
        ElementwiseCompute<functions::Inverse##name##Functor<T>, T>(           \
            dev_ctx, x, y, axis, functions::Inverse##name##Functor<T>(), out); \
      }                                                                        \
    }                                                                          \
  }

#define DEFINE_CUDA_ELEMENTWISE_OP(name)                                 \
  template <typename T, typename ContextT>                               \
  void name(const ContextT& dev_ctx,                                     \
            const DenseTensor& x,                                        \
            const DenseTensor& y,                                        \
            int axis,                                                    \
            DenseTensor* out) {                                          \
    std::vector<const DenseTensor*> inputs;                              \
    std::vector<DenseTensor*> outputs;                                   \
    inputs.emplace_back(&x);                                             \
    inputs.emplace_back(&y);                                             \
    outputs.emplace_back(out);                                           \
    out->mutable_data<T>();                                              \
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(         \
        dev_ctx, inputs, &outputs, axis, functions::name##Functor<T>()); \
  }
