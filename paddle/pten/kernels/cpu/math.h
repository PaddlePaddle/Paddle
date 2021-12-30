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
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

template <typename T>
void Mean(const CPUContext& dev_ctx,
          const DenseTensor& x,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          bool reduce_all,
          DenseTensor* out);

template <typename T>
void Add(const CPUContext& dev_ctx,
         const DenseTensor& x,
         const DenseTensor& y,
         int axis,
         DenseTensor* out);

template <typename T>
void Subtract(const CPUContext& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& y,
              int axis,
              DenseTensor* out);

template <typename T>
void Divide(const CPUContext& dev_ctx,
            const DenseTensor& x,
            const DenseTensor& y,
            int axis,
            DenseTensor* out);

template <typename T>
void Multiply(const CPUContext& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& y,
              int axis,
              DenseTensor* out);
template <typename T>
void Sum(const CPUContext& dev_ctx,
         const DenseTensor& x,
         const std::vector<int64_t>& dims,
         bool keep_dim,
         bool reduce_all,
         DataType out_dtype,
         DenseTensor* out);

}  // namespace pten

#define DEFINE_CPU_ELEMENTWISE_OP(name)                                      \
  template <typename T>                                                      \
  void name(const CPUContext& dev_ctx,                                       \
            const DenseTensor& x,                                            \
            const DenseTensor& y,                                            \
            int axis,                                                        \
            DenseTensor* out) {                                              \
    out->mutable_data<T>();                                                  \
    if (x.dims() == y.dims()) {                                              \
      SameDimsElementwiseCompute<                                            \
          general::SameDims##name##Functor<CPUContext, T>>()(                \
          dev_ctx, x, y, out);                                               \
    } else {                                                                 \
      auto x_dims = x.dims();                                                \
      auto y_dims = y.dims();                                                \
      if (x_dims.size() >= y_dims.size()) {                                  \
        ElementwiseCompute<general::name##Functor<T>, T>(                    \
            dev_ctx, x, y, axis, general::name##Functor<T>(), out);          \
      } else {                                                               \
        ElementwiseCompute<general::Inverse##name##Functor<T>, T>(           \
            dev_ctx, x, y, axis, general::Inverse##name##Functor<T>(), out); \
      }                                                                      \
    }                                                                        \
  }
