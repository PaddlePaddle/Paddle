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
#include "paddle/phi/core/selected_rows.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void LambKernel(const Context& dev_ctx,
                const DenseTensor& param,
                const SelectedRows& grad,
                const DenseTensor& learning_rate,
                const DenseTensor& moment1,
                const DenseTensor& moment2,
                const DenseTensor& beta1_pow,
                const DenseTensor& beta2_pow,
                const paddle::optional<DenseTensor>& master_param,
                const paddle::optional<DenseTensor>& skip_update,
                float weight_decay,
                float beta1,
                float beta2,
                float epsilon,
                bool always_adapt,
                bool multi_precision,
                DenseTensor* param_out,
                DenseTensor* moment1_out,
                DenseTensor* moment2_out,
                DenseTensor* beta1_pow_out,
                DenseTensor* beta2_pow_out,
                DenseTensor* master_param_outs);

}  // namespace sr
}  // namespace phi
