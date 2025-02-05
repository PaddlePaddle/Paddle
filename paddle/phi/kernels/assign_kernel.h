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

#include <vector>

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_array.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {

template <typename Context>
void AssignKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out);

template <typename Context>
DenseTensor Assign(const Context& dev_ctx, const DenseTensor& x) {
  DenseTensor out;
  MetaTensor meta_out(&out);
  MetaTensor meta_x(x);
  UnchangedInferMeta(meta_x, &meta_out);
  if (x.initialized()) {
    AssignKernel<Context>(dev_ctx, x, &out);
  }
  return out;
}

template <typename Context>
void Assign(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  MetaTensor meta_out(out);
  MetaTensor meta_x(x);
  UnchangedInferMeta(meta_x, &meta_out);
  if (x.initialized()) {
    AssignKernel<Context>(dev_ctx, x, out);
  }
}

// In order to be compatible with the `AsDispensable` input in the original
// assign op maker, the input parameter here needs to be dispensable, but
// this looks weird
template <typename Context>
void AssignRawKernel(const Context& dev_ctx,
                     const paddle::optional<DenseTensor>& x,
                     DenseTensor* out);

template <typename Context>
void AssignArrayKernel(const Context& dev_ctx,
                       const TensorArray& x,
                       TensorArray* out);

template <typename T, typename Context>
void AssignValueKernel(const Context& dev_ctx,
                       const std::vector<int>& shape,
                       DataType dtype,
                       const std::vector<Scalar>& values,
                       DenseTensor* out);

}  // namespace phi
