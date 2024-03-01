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

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/tensor_array.h"

namespace phi {
template <typename T, typename Context>
void CreateArrayKernel(const Context& dev_ctx,
                       DataType dtype,
                       TensorArray* out);

template <typename T, typename Context>
void CreateArrayLikeKernel(const Context& dev_ctx,
                           const TensorArray& input,
                           float val,
                           TensorArray* out);

template <typename T, typename Context>
void ArrayLengthKernel(const Context& dev_ctx,
                       const TensorArray& x,
                       DenseTensor* out);

template <typename T, typename Context>
void ArrayReadKernel(const Context& dev_ctx,
                     const TensorArray& array,
                     const Scalar& i,
                     DenseTensor* out);

template <typename T, typename Context>
void ArrayWriteKernel(const Context& dev_ctx,
                      const TensorArray& array,
                      const DenseTensor& x,
                      const Scalar& i,
                      TensorArray* out);

template <typename T, typename Context>
void ArrayToTensorKernel(const Context& dev_ctx,
                         const TensorArray& x,
                         int axis,
                         bool use_stack,
                         DenseTensor* out,
                         DenseTensor* out_index);

template <typename T, typename Context>
void ArrayPopKernel(const Context& dev_ctx,
                    const TensorArray& array,
                    int index,
                    TensorArray* array_out,
                    DenseTensor* out);

}  // namespace phi
