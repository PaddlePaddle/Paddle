// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/infermeta/multiary.h"
#include "paddle/pten/kernels/empty_kernel.h"
namespace pten {

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<DenseTensor>& x,
                  const Scalar& axis,
                  DenseTensor* out);

template <typename T, typename Context>
DenseTensor Concat(const Context& dev_ctx,
                   const std::vector<DenseTensor>& x,
                   const Scalar& axis) {
  std::vector<MetaTensor> meta_x;
  for (const auto& t : x) {
    meta_x.emplace_back(t);
  }

  auto dense_out = pten::Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  ConcatInferMeta(meta_x, axis.to<int>(), &meta_out, /*is_runtime=*/true);
  ConcatKernel<T, Context>(dev_ctx, x, axis, &dense_out);
  return dense_out;
}

}  // namespace pten
