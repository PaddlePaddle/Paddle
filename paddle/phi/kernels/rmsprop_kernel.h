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

template <typename T, typename Context>
void RmspropDenseKernel(const Context& dev_ctx,
                        const DenseTensor& param,
                        const DenseTensor& mean_square,
                        const DenseTensor& grad,
                        const DenseTensor& moment,
                        const DenseTensor& learning_rate,
                        paddle::optional<const DenseTensor&> mean_grad,
                        float epsilon,
                        float decay,
                        float momentum,
                        bool centered,
                        DenseTensor* param_out,
                        DenseTensor* moment_out,
                        DenseTensor* mean_square_out,
                        DenseTensor* mean_grad_out);

template <typename T, typename Context>
void RmspropSparseKernel(const Context& dev_ctx,
                         const DenseTensor& param,
                         const DenseTensor& mean_square,
                         const SelectedRows& grad,
                         const DenseTensor& moment,
                         const DenseTensor& learning_rate,
                         paddle::optional<const DenseTensor&> mean_grad,
                         float epsilon,
                         float decay,
                         float momentum,
                         bool centered,
                         DenseTensor* param_out,
                         DenseTensor* moment_out,
                         DenseTensor* mean_square_out,
                         DenseTensor* mean_grad_out);

}  // namespace phi
