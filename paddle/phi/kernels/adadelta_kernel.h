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
void AdadeltaKernel(const Context& dev_ctx,
                    const DenseTensor& param,
                    const DenseTensor& grad,
                    const DenseTensor& avg_squared_grad,
                    const DenseTensor& avg_squared_update,
                    const DenseTensor& learning_rate,
                    const paddle::optional<DenseTensor>& master_param,
                    float rho,
                    float epsilon,
                    bool multi_precision,
                    DenseTensor* param_out,
                    DenseTensor* avg_squared_grad_out,
                    DenseTensor* avg_squared_update_out,
                    DenseTensor* master_param_outs);

}  // namespace phi
