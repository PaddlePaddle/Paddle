// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
void RAdamKernel(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
                 const DenseTensor& learning_rate,
                 const DenseTensor& beta1_pow,
                 const DenseTensor& beta2_pow,
                 const DenseTensor& rho,
                 const DenseTensor& moment1,
                 const DenseTensor& moment2,
                 const paddle::optional<DenseTensor>& master_param,
                 float beta1,
                 float beta2,
                 float epsilon,
                 bool multi_precision,
                 DenseTensor* param_out,
                 DenseTensor* beta1_pow_out,
                 DenseTensor* beta2_pow_out,
                 DenseTensor* rho_out,
                 DenseTensor* moment1_out,
                 DenseTensor* moment2_out,
                 DenseTensor* master_param_out);
}  // namespace phi
