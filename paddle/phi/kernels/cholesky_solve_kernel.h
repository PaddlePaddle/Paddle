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
 * @brief Solves a linear system of equations with a positive semidefinite
 *        matrix to be inverted given its Cholesky factor matrix uu
 * @param  ctx     device context
 * @param  x       The input tensor, shape of (*,m,k)
 * @param  y       The input tensor, shape of (*,m,m) composed of upper or lower
 *                 triangular Cholesky factor
 * @param  upper   whether to consider the Cholesky factor as a lower or upper
 *                 triangular matrix
 * @param  out     The output tensor, shape same to x
 */
template <typename T, typename Context>
void CholeskySolveKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         bool upper,
                         DenseTensor* out);

}  // namespace phi
