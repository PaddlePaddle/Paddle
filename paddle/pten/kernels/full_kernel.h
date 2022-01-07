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

#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/dense_tensor.h"

#include "paddle/pten/infermeta/nullary.h"
#include "paddle/pten/kernels/empty_kernel.h"

namespace pten {

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const ScalarArray& shape,
                const Scalar& val,
                DenseTensor* out);

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const Scalar& val,
                    DenseTensor* out);

template <typename T, typename Context>
DenseTensor Full(const Context& dev_ctx,
                 const ScalarArray& shape,
                 const Scalar& val,
                 DataType dtype = DataType::FLOAT32,
                 Backend backend = Backend::CPU,  // Is backend needed here?
                 DataLayout layout = DataLayout::NCHW) {
  auto out_meta = CreateInferMeta(shape, dtype, layout);
  auto dense_out = Empty<T, Context>(dev_ctx, std::move(out_meta));
  FullKernel<T, Context>(dev_ctx, shape, val, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor FullLike(
    const Context& dev_ctx,
    const DenseTensor& x,
    const Scalar& val,
    DataType dtype = DataType::UNDEFINED,
    Backend backend = Backend::UNDEFINED,  // Is backend needed here?
    DataLayout layout = DataLayout::UNDEFINED) {
  auto out_meta = CreateLikeInferMeta(x.meta(), dtype, layout);
  auto dense_out = Empty<T, Context>(dev_ctx, std::move(out_meta));
  FullLikeKernel<T, Context>(dev_ctx, val, &dense_out);
  return dense_out;
}

}  // namespace pten
