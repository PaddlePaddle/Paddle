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
 * @brief This kernel is used to perform matrix vector multiplication
 *        of the input tensors `X` and `Vec`
 * @param  ctx   device context
 * @param  x     The matrix input of mv
 * @param  vec   The vector input of mv
 * @param  out   The output of mv
 */
template <typename T, typename Context>
void MvKernel(const Context& ctx,
              const DenseTensor& x,
              const DenseTensor& vec,
              DenseTensor* out);

}  // namespace phi
