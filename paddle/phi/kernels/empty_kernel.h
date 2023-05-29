// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {

template <typename T, typename Context>
void EmptyKernel(const Context& dev_ctx,
                 const IntArray& shape,
                 DataType dtype,
                 DenseTensor* out);

template <typename T, typename Context>
void EmptyLikeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     DataType dtype,
                     DenseTensor* out);

template <typename Context>
DenseTensor Empty(const Context& dev_ctx, DenseTensorMeta&& meta) {
  phi::DenseTensor dense_out;
  dense_out.set_meta(meta);
  dev_ctx.Alloc(&dense_out, dense_out.dtype());
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Empty(const Context& dev_ctx, const IntArray& shape) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  DataType dtype = phi::CppTypeToDataType<T>::Type();
  CreateInferMeta(shape, dtype, &meta_out);
  EmptyKernel<T, Context>(dev_ctx, shape, dtype, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor EmptyLike(const Context& dev_ctx, const DenseTensor& x) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  DataType dtype = phi::CppTypeToDataType<T>::Type();
  CreateLikeInferMeta(x, dtype, &meta_out);
  EmptyLikeKernel<T, Context>(dev_ctx, x, dtype, &dense_out);
  return dense_out;
}

}  // namespace phi
