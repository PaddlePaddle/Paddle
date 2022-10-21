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

template <typename T,typename Context>
void MultiTensorAdamKernel(const Context& dev_ctx,
                           const std::vector<const DenseTensor *> &params,
                           const std::vector<const DenseTensor *> &grads,
                           const std::vector<const DenseTensor *> &moments1,
                           const std::vector<const DenseTensor *> &moments2,
                           const paddle::optional<std::vector<const DenseTensor*>>& master_param,
                           const DenseTensor &beta1_pow,
                           const DenseTensor &beta2_pow,
                           const DenseTensor &learning_rate,
                           const paddle::optional<DenseTensor>& skip_update,
                           const Scalar& beta1,
                           const Scalar& beta2,
                           const Scalar& epsilon,
                           int chunk_size,
                           float weight_decay,
                           bool mode,
                           bool multi_precision,
                           bool use_global_beta_pow,
                           std::vector<DenseTensor *> params_out,
                           std::vector<DenseTensor *> moments1_out,
                           std::vector<DenseTensor *> moments2_out,
                           std::vector<DenseTensor *> master_param_out,
                           DenseTensor *beta1_pow_out,
                           DenseTensor *beta2_pow_out);

}  // namespace phi
