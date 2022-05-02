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

template <typename T, typename Context>
void AsgdKernel(const Context& dev_ctx,
                const DenseTensor& param,
                const DenseTensor& learning_rate,
                const DenseTensor& grad,
                const DenseTensor& avg_param,
                const DenseTensor& current_step,
                float t0,
                DenseTensor* param_out,
                DenseTensor* avg_param_out,
                DenseTensor* current_step_out);

}  // namespace phi
