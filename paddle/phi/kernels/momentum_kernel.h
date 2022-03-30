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
void MomentumDenseKernel(const Context& dev_ctx,
                         const DenseTensor& param,
                         const DenseTensor& grad,
                         const DenseTensor& velocity,
                         const DenseTensor& learning_rate,
                         paddle::optional<const DenseTensor&> master_param,
                         float mu,
                         bool use_nesterov,
                         const std::string& regularization_method,
                         float regularization_coeff,
                         bool multi_precision,
                         float rescale_grad,
                         DenseTensor* param_out,
                         DenseTensor* velocity_out,
                         DenseTensor* master_param_out);

template <typename T, typename Context>
void MomentumSparseKernel(const Context& dev_ctx,
                          const DenseTensor& param,
                          const SelectedRows& grad,
                          const DenseTensor& velocity,
                          const DenseTensor& learning_rate,
                          paddle::optional<const DenseTensor&> master_param,
                          float mu,
                          bool use_nesterov,
                          const std::string& regularization_method,
                          float regularization_coeff,
                          bool multi_precision,
                          float rescale_grad,
                          DenseTensor* param_out,
                          DenseTensor* velocity_out,
                          DenseTensor* master_param_out);

}  // namespace phi
