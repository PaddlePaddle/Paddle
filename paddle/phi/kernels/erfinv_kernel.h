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
 * @brief This kernel is used to compute inverse error function of x.
 *
 *        The equation is:
 *        $$erfinv(x) = {ndtri({x \over 2} + 0.5)} \over {\sqrt{2}}$$
 *
 *        The input `x` can carry the LoD (Level of Details) information,
 *        or not. And the output shares the LoD information with `x`
 * @param  ctx     device context
 * @param  x       the input tensor of erfinv
 * @param  out     the output tensor of erfinv
 */
template <typename T, typename Context>
void ErfinvKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out);

}  // namespace phi
