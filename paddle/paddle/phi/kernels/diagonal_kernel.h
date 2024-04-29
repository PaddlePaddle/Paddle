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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {

/**
 * @brief Return a partial view of input with the its diagonal elements
 *        of the input tensor. The behavior of this operator is similar to
 *        how `numpy.diagonal` works.
 * @param  ctx     device context
 * @param  x       the input tensor, from which the diagonals are taken
 * @param  offset  offset of the diagonal from the main diagonal. Can be both
 *                 positive and negative
 * @param  axis1   the first axis of the 2-D planes from which the diagonals
 *                 should be taken. Can be either positive or negative
 * @param  axis2   the second axis of the 2-D planes from which the diagonals
 *                 should be taken. Can be either positive or negative
 * @param  out     the partial view of input with the its diagonal elements
 */
template <typename T, typename Context>
void DiagonalKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    int offset,
                    int axis1,
                    int axis2,
                    DenseTensor* out);

template <typename Context>
void DiagonalStridedKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           int offset,
                           int axis1,
                           int axis2,
                           DenseTensor* out);

template <typename T, typename Context>
DenseTensor Diagonal(const Context& dev_ctx,
                     const DenseTensor& x,
                     int offset,
                     int axis1,
                     int axis2) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  DiagonalInferMeta(x, offset, axis1, axis2, &meta_out);
  DiagonalKernel<T, Context>(dev_ctx, x, offset, axis1, axis2, &dense_out);
  return dense_out;
}
}  // namespace phi
