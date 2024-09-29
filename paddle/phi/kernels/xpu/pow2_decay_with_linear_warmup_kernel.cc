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

#include "paddle/phi/kernels/pow2_decay_with_linear_warmup_kernel.h"

#include "paddle/common/macros.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void Pow2DecayWithLinearWarmupKernel(const Context& dev_ctx,
                                     const DenseTensor& lr,
                                     const DenseTensor& step,
                                     int64_t warmup_steps,
                                     int64_t total_steps,
                                     float base_lr,
                                     float end_lr,
                                     DenseTensor* lr_out,
                                     DenseTensor* step_out) {
  PADDLE_ENFORCE_EQ(&lr,
                    lr_out,
                    common::errors::InvalidArgument("Input(LearningRate) and "
                                                    "Output(LearningRateOut) "
                                                    "must be the same."));
  PADDLE_ENFORCE_EQ(&step,
                    step_out,
                    common::errors::InvalidArgument(
                        "Input(Step) and Output(StepOut) must be the same."));
  PADDLE_ENFORCE_EQ(
      step.initialized(),
      true,
      common::errors::InvalidArgument("Input(Step) must be initialized."));

  PADDLE_ENFORCE_LE(warmup_steps,
                    total_steps,
                    common::errors::InvalidArgument(
                        "warmup_steps must not be larger than total_steps."));

  auto* lr_data = lr_out->data<T>();
  auto* step_data = step_out->data<int64_t>();
  int r = xpu::pow2_decay_with_linear_warmup(dev_ctx.x_context(),
                                             lr_data,
                                             step_data,
                                             static_cast<size_t>(warmup_steps),
                                             static_cast<size_t>(total_steps),
                                             base_lr,
                                             end_lr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pow2_decay_with_linear_warmup");
}

}  // namespace phi

PD_REGISTER_KERNEL(pow2_decay_with_linear_warmup,
                   XPU,
                   ALL_LAYOUT,
                   phi::Pow2DecayWithLinearWarmupKernel,
                   float) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
