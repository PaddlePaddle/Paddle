/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

/**
 * @brief Computes the Cholesky decomposition of one symmetric positive-definite
 *        matrix or batches of symmetric positive-definite matrices.
 * @param  ctx     device context
 * @param  x       The input tensor of cholesky op. Its shape should be
 *                 [*, M, M] where * is zero or more batch dimensions,
 *                 and matrices on the inner-most 2 dimensions all
 *                 should be symmetric positive-definite
 * @param  upper   flag indicating whether to return upper or lower triangular
 *                 matrices
 * @param  out     The output tensor of cholesky kernel. It has the same
 *                 shape as the input, and it is composed of upper-triangular or
 *                 lower-triangular Cholesky factors of each of the individual
 *                 matrices
 */
template <typename T, typename Context>
void CholeskyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    bool upper,
                    DenseTensor* out);

}  // namespace phi
