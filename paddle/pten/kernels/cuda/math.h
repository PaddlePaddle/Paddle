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

#include "paddle/pten/common/scalar.h"
#include "paddle/pten/core/dense_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"

namespace pten {

using CUDAContext = paddle::platform::CUDADeviceContext;

template <typename T>
void Sign(const CUDAContext& dev_ctx, const DenseTensor& x, DenseTensor* out);

template <typename T>
void Mean(const CUDAContext& dev_ctx,
          const DenseTensor& x,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          bool reduce_all,
          DataType in_dtype,
          DataType out_dtype,
          DenseTensor* out);

template <typename T>
void Scale(const CUDAContext& dev_ctx,
           const DenseTensor& x,
           const Scalar& scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out);

template <typename T>
void ElementwiseAdd(const CUDAContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out);

template <typename T>
void ElementwiseSub(const CUDAContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out);

template <typename T>
void ElementwiseDiv(const CUDAContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out);

template <typename T>
void ElementwiseMul(const CUDAContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out);

template <typename T>
void Sum(const CUDAContext& dev_ctx,
         const DenseTensor& x,
         const std::vector<int64_t>& dims,
         bool keep_dim,
         bool reduce_all,
         DataType in_dtype,
         DataType out_dtype,
         DenseTensor* out);

}  // namespace pten

#define DEFINE_CUDA_ELEMENTWISE_OP(name)                               \
  template <typename T>                                                \
  void Elementwise##name(const CUDAContext& dev_ctx,                   \
                         const DenseTensor& x,                         \
                         const DenseTensor& y,                         \
                         int axis,                                     \
                         DenseTensor* out) {                           \
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
