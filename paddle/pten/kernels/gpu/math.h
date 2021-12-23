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

// CUDA and HIP use same api
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {

template <typename T>
void Mean(const GPUContext& dev_ctx,
          const DenseTensor& x,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          bool reduce_all,
          DenseTensor* out);

template <typename T>
void Add(const GPUContext& dev_ctx,
         const DenseTensor& x,
         const DenseTensor& y,
         int axis,
         DenseTensor* out);

template <typename T>
void Subtract(const GPUContext& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& y,
              int axis,
              DenseTensor* out);

template <typename T>
void Divide(const GPUContext& dev_ctx,
            const DenseTensor& x,
            const DenseTensor& y,
            int axis,
            DenseTensor* out);

template <typename T>
void Multiply(const GPUContext& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& y,
              int axis,
              DenseTensor* out);

template <typename T>
void Sum(const GPUContext& dev_ctx,
         const DenseTensor& x,
         const std::vector<int64_t>& dims,
         bool keep_dim,
         bool reduce_all,
         DataType out_dtype,
         DenseTensor* out);

}  // namespace pten

#define DEFINE_CUDA_ELEMENTWISE_OP(name)                               \
  template <typename T>                                                \
  void name(const GPUContext& dev_ctx,                                 \
            const DenseTensor& x,                                      \
            const DenseTensor& y,                                      \
            int axis,                                                  \
            DenseTensor* out) {                                        \
    std::vector<const DenseTensor*> inputs;                            \
    std::vector<DenseTensor*> outputs;                                 \
    inputs.emplace_back(&x);                                           \
    inputs.emplace_back(&y);                                           \
    outputs.emplace_back(out);                                         \
    out->mutable_data<T>();                                            \
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(       \
        dev_ctx, inputs, &outputs, axis, general::name##Functor<T>()); \
  }

#endif
