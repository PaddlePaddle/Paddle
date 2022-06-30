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
 * @brief This kernrel is used to perform elementwise digamma for x.
 *        $$out = \Psi(x) = \frac{ \Gamma^{'}(x) }{ \Gamma(x) }$$
 * @param  ctx     device context
 * @param  x       the input tensor of digamma
 * @param  out     the output tensor of digamma
 */
template <typename T, typename Context>
void DigammaKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out);

}  // namespace phi
