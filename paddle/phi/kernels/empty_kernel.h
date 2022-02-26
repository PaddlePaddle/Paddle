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

#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {

template <typename T, typename Context>
void EmptyKernel(const Context& dev_ctx,
                 const ScalarArray& shape,
                 DataType dtype,
                 DenseTensor* out);

template <typename T, typename Context>
void EmptyLikeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     DataType dtype,
                     DenseTensor* out);

// TODO(chenweihang): the tensor creation method need to be replaced later,
// all kernel api call Empty here instead of making tensor self
template <typename Context>
DenseTensor Empty(const Context& dev_ctx, DenseTensorMeta&& meta) {
  phi::DenseTensor dense_out(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(meta));
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Empty(const Context& dev_ctx) {
  return Empty(dev_ctx,
               {paddle::experimental::CppTypeToDataType<T>::Type(),
                {-1},
                DataLayout::NCHW});
}

template <typename T, typename Context>
DenseTensor Empty(const Context& dev_ctx, const ScalarArray& shape) {
  auto dense_out = Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  DataType dtype = paddle::experimental::CppTypeToDataType<T>::Type();
  CreateInferMeta(shape, dtype, &meta_out);
  EmptyKernel<T, Context>(dev_ctx, shape, dtype, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor EmptyLike(const Context& dev_ctx, const DenseTensor& x) {
  auto dense_out = Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  DataType dtype = paddle::experimental::CppTypeToDataType<T>::Type();
  CreateLikeInferMeta(x, dtype, &meta_out);
  EmptyLikeKernel<T, Context>(dev_ctx, x, dtype, &dense_out);
  return dense_out;
}

}  // namespace phi
