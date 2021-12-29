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
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/include/infermeta.h"

namespace pten {

template <typename T, typename Context>
void FullKernel(const Context& context,
                const ScalarArray& shape,
                const Scalar& val,
                DenseTensor* out);

template <typename T, typename Context>
void FullLikeKernel(const Context& context,
                    const Scalar& val,
                    DenseTensor* out);

template <typename T, typename Context>
DenseTensor Full(const Context& context,
                 const ScalarArray& shape,
                 const Scalar& val,
                 DataType dtype = DataType::FLOAT32,
                 Backend backend = Backend::CPU,  // Is backend needed here?
                 DataLayout layout = DataLayout::NCHW) {
  auto out_meta = CreateInferMeta(shape, dtype, layout);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          context.GetPlace()),
      std::move(out_meta));
  FullKernel<T, Context>(context, shape, val, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor FullLike(
    const Context& context,
    const DenseTensor& x,
    const Scalar& val,
    DataType dtype = DataType::UNDEFINED,
    Backend backend = Backend::UNDEFINED,  // Is backend needed here?
    DataLayout layout = DataLayout::UNDEFINED) {
  auto out_meta = CreateLikeInferMeta(x.meta(), dtype, layout);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          context.GetPlace()),
      std::move(out_meta));
  FullLikeKernel<T, Context>(context, val, &dense_out);
  return dense_out;
}

}  // namespace pten
