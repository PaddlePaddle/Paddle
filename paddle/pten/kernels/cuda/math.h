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

#include "paddle/pten/core/dense_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"

namespace pten {

using CUDAContext = paddle::platform::CUDADeviceContext;

template <typename T>
void Sign(const CUDAContext& dev_ctx, const DenseTensor& x, DenseTensor* out);

template <typename T>
void Mean(const CUDAContext& dev_ctx, const DenseTensor& x, DenseTensor* out);

template <typename T>
void Scale(const CUDAContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out);

template <typename T>
void ScaleHost(const CUDAContext& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& scale,
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

}  // namespace pten

#endif
