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

#include "paddle/pten/core/dense_tensor.h"

#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/empty_kernel.h"

namespace pten {

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const ScalarArray& num_or_sections,
                 const Scalar& axis,
                 std::vector<DenseTensor*> out);

template <typename T, typename Context>
std::vector<DenseTensor> Split(const Context& dev_ctx,
                               const DenseTensor& x,
                               const ScalarArray& num_or_sections,
                               const Scalar& axis) {
  std::vector<MetaTensor> out_meta;
  SplitInferMeta(x, num_or_sections, axis, &out_meta, true);

  std::vector<DenseTensor> result;
  result.reserve(out_meta.size());
  for (size_t i = 0; i < out_meta.size(); ++i) {
    auto dense_out = pten::Empty<T, Context>(dev_ctx, std::move(out_meta[i]));
    result.push_back(dense_out);
  }

  std::vector<DenseTensor*> outs;
  outs.reserve(out_meta.size());
  for (size_t i = 0; i < out_meta.size(); ++i) {
    outs.push_back(&result[i]);
  }

  SplitKernel<T, Context>(dev_ctx, x, num_or_sections, axis, outs);

  return result;
}

}  // namespace pten
