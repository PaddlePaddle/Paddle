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
 * @brief Given two tensors x and y, compute Lp-norm of (x-y).
 *        It is not a norm in a strict sense, only as a measure of distance.
 *        The shapes of x and y must be broadcastable. Where, z = x - y,
 *
 *        When p = 0, defining $0^0 = 0$, the zero-norm of z is simply
 *        the number of non-zero elements of z.
 *        $$
 *        ||z||_{0} = \lim_{p \rightarrow 0} \sum_{i=1}^{m} |z_i|^p
 *        $$
 *
 *        When p = inf, the inf-norm of z is the maximum element of z.
 *        $$
 *        ||z||_\infty=\max_i |z_i|
 *        $$
 *
 *        When p = -inf, the negative-inf-norm of z is the minimum element of z.
 *        $$
 *        ||z||_{-\infty}=\min_i |z_i|
 *        $$
 *
 *        Otherwise, the p-norm of z follows the formula,
 *        $$
 *        ||z||_{p} = (\sum_{i=i}^{m} |z_i|^p)^{1/p}
 *        $$
 * @param  ctx     device context
 * @param  x       the input Tensor of Dist
 * @param  y       the Right-hand-side input Tensor of Dist
 * @param  p       the norm to be computed
 * @param  out     the output of Dist, which is the p-norm of (x - y)
 */
template <typename T, typename Context>
void DistKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                float p,
                DenseTensor* out);

}  // namespace phi
