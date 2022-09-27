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
 * @brief If ``x`` is a vector (1-D tensor), a 2-D square tensor with the
 *        elements of ``x`` as the diagonal is returned.
 *        If ``x`` is a matrix (2-D tensor), a 1-D tensor with the diagonal
 *        elements of ``x`` is returned.
 *
 *        The argument ``offset`` controls the diagonal offset:
 *        If ``offset`` = 0, it is the main diagonal.
 *        If ``offset`` > 0, it is superdiagonal. If ``offset`` < 0,
 *        it is subdiagonal.
 * @param  ctx             device context
 * @param  x               The input tensor. Its shape is either 1-D or 2-D.
 * @param  offset          The diagonal offset. A positive value represents
 *                         superdiagonal, 0 represents the main diagonal, and a
 *                         negative value represents subdiagonal.
 * @param  padding_value   Use this value to fill the area outside the specified
 *                         diagonal band. Only takes effect when the input is a
 *                         1-D Tensor. The default value is 0.
 * @param  out             The output tensor. A square matrix or a vector.
 */
template <typename T, typename Context>
void DiagKernel(const Context& dev_ctx,
                const DenseTensor& x,
                int offset,
                float padding_value,
                DenseTensor* out);

template <typename T, typename Context>
DenseTensor Diag(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 float padding_value) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  DiagInferMeta(x, offset, padding_value, &meta_out);
  DiagKernel<T, Context>(dev_ctx, x, offset, padding_value, &dense_out);
  return dense_out;
}

}  // namespace phi
