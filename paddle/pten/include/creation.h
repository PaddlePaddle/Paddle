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
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/kernels/empty_kernel.h"
#include "paddle/pten/kernels/full_kernel.h"

namespace pten {

// TODO(YuanRisheng) This function name should be same as User API name.
// TODO(zyfncg) Automatic code generation
template <typename T, typename ContextT>
DenseTensor Empty(const ContextT& dev_ctx,
                  const ScalarArray& shape,
                  DataType dtype = DataType::FLOAT32,
                  Backend backend = Backend::CPU,  // Is backend needed here?
                  DataLayout layout = DataLayout::NCHW) {
  auto out_meta = CreateInferMeta(shape, dtype, layout);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Empty<T, ContextT>(dev_ctx, shape, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor EmptyLike(
    const ContextT& dev_ctx,
    const DenseTensor& x,
    DataType dtype = DataType::UNDEFINED,
    Backend backend = Backend::UNDEFINED,  // Is backend needed here?
    DataLayout layout = DataLayout::UNDEFINED) {
  auto out_meta = CreateLikeInferMeta(x.meta(), dtype, layout);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  EmptyLike<T, ContextT>(dev_ctx, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Full(const ContextT& dev_ctx,
                 const ScalarArray& shape,
                 const Scalar& val,
                 DataType dtype = DataType::FLOAT32,
                 Backend backend = Backend::CPU,  // Is backend needed here?
                 DataLayout layout = DataLayout::NCHW) {
  auto out_meta = CreateInferMeta(shape, dtype, layout);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Full<T, ContextT>(dev_ctx, shape, val, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor FullLike(
    const ContextT& dev_ctx,
    const DenseTensor& x,
    const Scalar& val,
    DataType dtype = DataType::UNDEFINED,
    Backend backend = Backend::UNDEFINED,  // Is backend needed here?
    DataLayout layout = DataLayout::UNDEFINED) {
  auto out_meta = CreateLikeInferMeta(x.meta(), dtype, layout);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  FullLike<T, ContextT>(dev_ctx, val, &dense_out);
  return dense_out;
}

}  // namespace pten
