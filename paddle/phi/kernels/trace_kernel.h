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
 * @brief Trace Kernel.
 *        Return the sum along diagonals of the input tensor.
 *        The behavior of this operator is similar to how `numpy.trace` works.
 *
 *        If Input is 2-D, returns the sum of diagonal.
 *        If Input has larger dimensions, then returns an tensor of diagonals
 *        sum, diagonals be taken from the 2-D planes specified by dim1 and
 *        dim2.
 * @param  ctx      device context
 * @param  x        The input tensor, from which the diagonals are taken
 * @param  offset   offset of the diagonal from the main diagonal.
 *                  Can be bothpositive and negative.
 * @param  axis1    the first axis of the 2-D planes from which the diagonals
 *                  should be taken. Can be either positive or negative
 * @param  axis2    the second axis of the 2-D planes from which the diagonals
 *                  should be taken. Can be either positive or negative
 * @param  out      the sum along diagonals of the input tensor
 */
template <typename T, typename Context>
void TraceKernel(const Context& ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out);

}  // namespace phi
