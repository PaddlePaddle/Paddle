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
#include "paddle/phi/core/selected_rows.h"

namespace phi {

template <typename T, typename Context>
void RmsNormGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const paddle::optional<DenseTensor>& bias,
                       const paddle::optional<DenseTensor>& residual,
                       const DenseTensor& norm_weight,
                       const paddle::optional<DenseTensor>& norm_bias,
                       const DenseTensor& inv_var,
                       const DenseTensor& out_grad,
                       const float epsilon,
                       const int begin_norm_axis,
                       const float quant_scale,
                       DenseTensor* x_grad,
                       DenseTensor* norm_weight_grad,
                       DenseTensor* norm_bias_grad);

}  // namespace phi
