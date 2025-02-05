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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out);

template <typename Context>
void TransposeStridedKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const std::vector<int>& axis,
                            DenseTensor* out);

template <typename T, typename Context>
void Transpose(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int>& axis,
               DenseTensor* dense_out) {
  MetaTensor meta_out(dense_out);
  TransposeInferMeta(x, axis, &meta_out);

  // do not call TransposeStridedKernel, because some other kernels call
  // Transpose directly
  if (x.has_allocation()) {
    TransposeKernel<T, Context>(dev_ctx, x, axis, dense_out);
  }
}

template <typename T, typename Context>
DenseTensor Transpose(const Context& dev_ctx,
                      const DenseTensor& x,
                      const std::vector<int>& axis) {
  DenseTensor dense_out;
  Transpose<T, Context>(dev_ctx, x, axis, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor TransposeLast2Dim(const Context& dev_ctx, const DenseTensor& x) {
  size_t rank = x.dims().size();
  std::vector<int> axis(rank);
  for (size_t i = 0; i < rank; ++i) {
    axis[i] = i;
  }
  std::swap(axis[rank - 1], axis[rank - 2]);
  return Transpose<T, Context>(dev_ctx, x, axis);
}

}  // namespace phi
