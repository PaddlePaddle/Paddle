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
 * @brief Erf Kernel.
 *        The equation is:
 *        $$
 *        f(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x}e^{- \eta^{2}}d\eta
 *        $$
 *
 *        The input `x` can carry the LoD (Level of Details) information,
 *        or not. And the output shares the LoD information with input `x`.
 * @param  ctx   device context
 * @param  x     The input tensor of erf kernel
 * @param  out   The output tensor of erf kernel
 */
template <typename T, typename Context>
void ErfKernel(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out);

}  // namespace phi
