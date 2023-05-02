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
 * @brief Returns the cross product of vectors in dimension dim of
 *        input and other. Input and other must have the same size,
 *        and the size of their dim dimension should be 3.
 *        If dim is not given, it defaults to the first dimension
 *        found with the size 3.
 * @param  ctx     device context
 * @param  x       the input tensor
 * @param  y       the second input tensor
 * @param  axis    the dimension to take the cross-product in
 * @param  out     the output tensor
 */
template <typename T, typename Context>
void CrossKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 int axis,
                 DenseTensor* out);

}  // namespace phi
