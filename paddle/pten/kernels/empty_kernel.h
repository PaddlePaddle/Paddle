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

#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/infermeta/nullary.h"
#include "paddle/pten/infermeta/unary.h"

namespace pten {

template <typename T, typename Context>
void EmptyKernel(const Context& dev_ctx,
                 const ScalarArray& shape,
                 DenseTensor* out);

template <typename T, typename Context>
void EmptyLikeKernel(const Context& dev_ctx, DenseTensor* out);

// TODO(chenweihang): the tensor creation method need to be replaced later,
// all kernel api call Empty here instead of making tensor self
template <typename T, typename Context>
DenseTensor Empty(const Context& dev_ctx, DenseTensorMeta&& meta) {
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(meta));
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Empty(const Context& dev_ctx) {
  return Empty<T, Context>(dev_ctx,
                           {paddle::experimental::CppTypeToDataType<T>::Type(),
                            {-1},
                            DataLayout::NCHW});
}

template <typename T, typename Context>
DenseTensor Empty(const Context& dev_ctx,
                  const ScalarArray& shape,
                  DataType dtype = DataType::FLOAT32,
                  Backend backend = Backend::CPU,  // Is backend needed here?
                  DataLayout layout = DataLayout::NCHW) {
  auto out_meta = CreateInferMeta(shape, dtype, layout);
  auto dense_out = Empty<T, Context>(dev_ctx, std::move(out_meta));
  EmptyKernel<T, Context>(dev_ctx, shape, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor EmptyLike(
    const Context& dev_ctx,
    const DenseTensor& x,
    DataType dtype = DataType::UNDEFINED,
    Backend backend = Backend::UNDEFINED,  // Is backend needed here?
    DataLayout layout = DataLayout::UNDEFINED) {
  auto out_meta = CreateLikeInferMeta(x.meta(), dtype, layout);
  auto dense_out = Empty<T, Context>(dev_ctx, std::move(out_meta));
  EmptyLikeKernel<T, Context>(dev_ctx, &dense_out);
  return dense_out;
}

}  // namespace pten
