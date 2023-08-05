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

#include "paddle/phi/kernels/distributed_fused_lamb_init_op.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void DistributedFusedLambInitOpKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& param,
    const std::vector<const DenseTensor*>& grad,
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
    const std::vector<DenseTensor*>& param_out,
    const std::vector<DenseTensor*>& master_param_out,
    const std::vector<DenseTensor*>& grad_out,
    DenseTensor* global_scale,
    DenseTensor* step,
    float beta1,
    float beta2,
    std::vector<int> apply_weight_decay,
    int alignment,
    int rank,
    int nranks) {
  PADDLE_THROW(phi::errors::Unavailable(
      "Do not support expert count op for cpu kernel now."));
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(distributed_fused_lamb_init_op,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::DistributedFusedLambInitOpKernel,
                   float) {}
