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

namespace phi {

/**
 * @brief Performs sorting on the input tensor along the given axis and outputs
 *        two tensors, Output(Out) and Output(Indices). They reserve the same
 *        shape with Input(X), and Output(Out) represents the sorted tensor
 *        while Output(Indices) gives the sorted order along the given axis
 *        Attr(axis).
 * @param  ctx          device context
 * @param  x            The input of Argsort
 * @param  axis         The axis along which to sort the tensor.
 *                      When axis < 0, the actual axis will be the |axis|'th
 *                      counting backwards
 * @param  descending   The descending attribute is a flag to tell
 *                      algorithm how to sort the input data.
 *                      If descending is true, will sort by descending order,
 *                      else if false, sort by ascending order
 * @param  stable       Indicate whether to use stable sorting algorithm, which
 *                      guarantees that the order of equivalent elements is
 *                      preserved.
 * @param  out          The sorted tensor of Argsort op, with the same shape as
 *                      x
 * @param  indices      The indices of a tensor giving the sorted order, with
 *                      the same shape as x
 */
template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   int axis,
                   bool descending,
                   bool stable,
                   DenseTensor* output,
                   DenseTensor* indices);

}  // namespace phi
