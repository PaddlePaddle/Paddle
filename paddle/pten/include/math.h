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

// See Note: [ How do we organize the kernel directory ]
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/kernels/complex_kernel.h"
#include "paddle/pten/kernels/scale_kernel.h"

namespace pten {

template <typename T, typename ContextT>
DenseTensor Sign(const ContextT& dev_ctx, const DenseTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Sign<T>(dev_ctx, x, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Scale(const ContextT& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& scale,
                  float bias,
                  bool bias_after_scale) {
  auto out_meta = UnchangedInferMeta(x.meta());
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Scale<T, ContextT>(dev_ctx, x, scale, bias, bias_after_scale, &dense_out);
  return dense_out;
}

template <typename T, typename ContextT>
DenseTensor Conj(const ContextT& dev_ctx, const DenseTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  pten::DenseTensor dense_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  Conj<T>(dev_ctx, x, &dense_out);
  return dense_out;
}

}  // namespace pten
