// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void DistributedFusedLambInitOpKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& param,
    const std::vector<const DenseTensor*>& grad,
    float beta1,
    float beta2,
    const std::vector<int>& apply_weight_decay,
    int alignment,
    int rank,
    int nranks,
    DenseTensor* fp32_fused_param,
    DenseTensor* fp32_fused_grad,
    DenseTensor* fp16_fused_param,
    DenseTensor* fp16_fused_grad,
    DenseTensor* moment1,
    DenseTensor* moment2,
    DenseTensor* beta1_pow,
    DenseTensor* beta2_pow,
    DenseTensor* fused_param_offsets,
    DenseTensor* fp32_shard_fused_param_offsets,
    DenseTensor* fp16_shard_fused_param_offsets,
    DenseTensor* param_info,
    DenseTensor* param_order,
    std::vector<DenseTensor*> param_out,
    std::vector<DenseTensor*> master_param_out,
    std::vector<DenseTensor*> grad_out,
    DenseTensor* global_scale,
    DenseTensor* step);

}  // namespace phi
