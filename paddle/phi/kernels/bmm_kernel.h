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
 * @brief Bmm Kernel.
 *        Applies batched matrix multiplication to two tensors.
 *
 *        Both of the two input tensors must be three-dimensional
 *        and share the same batch size.
 *        if x is a (b, m, k) tensor, y is a (b, k, n) tensor,
 *        the output will be a (b, m, n) tensor.
 *
 * @param  ctx      device context
 * @param  x        The input tensor
 * @param  y        The input tensor
 * @param  out      The product Tensor
 */
template <typename T, typename Context>
void BmmKernel(const Context& ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out);

}  // namespace phi
