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
#include "paddle/phi/core/device_context.h"

namespace phi {

/**
 * @brief This kernel generate random value that follow binomial distribution.
 * @param  ctx          device context
 * @param  total_count  A Tensor with each element inidicating the number of
 * bernoulli experiments
 * @param  prob         A Tensor with each element inidicating probability of
 * success for one bernoulli experiment
 * @param  out          A Tensor filled with returned random value
 */
template <typename T, typename Context>
void BinomialiKernel(const Context& ctx,
                     const DenseTensor& total_count,
                     const DenseTensor& prob,
                     DenseTensor* out);

}  // namespace phi
