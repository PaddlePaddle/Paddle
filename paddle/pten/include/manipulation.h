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

// See Note: [ How do we organize the kernel directory ]
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/kernels/cast_kernel.h"
#include "paddle/pten/kernels/flatten_kernel.h"
#include "paddle/pten/kernels/reshape_kernel.h"

namespace pten {

template <typename T, typename ContextT>
DenseTensor Flatten(const ContextT& dev_ctx,
                    const DenseTensor& x,
                    int start_axis,
                    int stop_axis) {
  auto out_meta = FlattenInferMeta(x.meta(), start_axis, stop_axis);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Flatten<T, ContextT>(dev_ctx, x, start_axis, stop_axis, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Cast(const ContextT& dev_ctx,
                 const DenseTensor& x,
                 DataType out_dtype) {
  auto out_meta = CastInferMeta(x.meta(), out_dtype);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Cast<T, ContextT>(dev_ctx, x, out_dtype, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Reshape(const ContextT& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<int64_t>& shape) {
  auto out_meta = InferMetaFromVecValue(x.meta(), shape);
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Reshape<ContextT>(dev_ctx, x, ScalarArray(shape), &dense_out);
  return dense_out;
}

}  // namespace pten
