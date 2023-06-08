// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/device_context.h"

namespace phi {

/**
 * @brief This kernel is used to perform elementwise polygamma for x.
 * @param  ctx     device context
 * @param  x       the input tensor of polygamma
 * @param  n       the input tensor of polygamma
 * @param  out     the output tensor of polygamma
 */
template <typename T, typename Context>
void PolygammaKernel(const Context& ctx,
                     const DenseTensor& x,
                     const int n,
                     DenseTensor* out);

}  // namespace phi
