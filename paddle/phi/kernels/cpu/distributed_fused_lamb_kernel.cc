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

#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void DistributedFusedLambKernel(const Context &dev_ctx,
                                const std::vector<const DenseTensor *> &param,
                                const std::vector<const DenseTensor *> &grad,
                                const paddle::optional<DenseTensor> &fp32_param,
                                const paddle::optional<DenseTensor> &fp32_grad,
                                const paddle::optional<DenseTensor> &fp16_param,
                                const paddle::optional<DenseTensor> &fp16_grad,
                                const DenseTensor &moment1,
                                const DenseTensor &moment2,
                                const DenseTensor &beta1_pow,
                                const DenseTensor &beta2_pow,
                                const DenseTensor &param_offsets,
                                const DenseTensor &fp32_partial_offsets,
                                const DenseTensor &fp16_partial_offsets,
                                const DenseTensor &param_info,
                                const DenseTensor &param_order,
                                const DenseTensor &learning_rate,
                                const DenseTensor &global_scale,
                                int acc_steps,
                                float beta1,
                                float beta2,
                                float epsilon,
                                float max_global_grad_norm,
                                float weight_decay,
                                bool clip_after_allreduce,
                                bool use_master_param_norm,
                                bool use_master_acc_grad,
                                bool is_grad_scaled_by_nranks,
                                bool use_hierarchical_allreduce,
                                int64_t nranks,
                                const std::vector<int> &ring_ids,
                                DenseTensor *fp32_param_out,
                                DenseTensor *fp16_param_out,
                                DenseTensor *fp32_acc_grad,
                                DenseTensor *fp16_acc_grad,
                                DenseTensor *moment1_out,
                                DenseTensor *moment2_out,
                                DenseTensor *beta1_pow_out,
                                DenseTensor *beta2_pow_out,
                                DenseTensor *param_out,
                                DenseTensor *found_inf,
                                DenseTensor *acc_step,
                                DenseTensor *stop_update,
                                DenseTensor *step) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "The distributed_fused_lamb operator does not support CPU yet."));
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(distributed_fused_lamb,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::DistributedFusedLambKernel,
                   float) {}
