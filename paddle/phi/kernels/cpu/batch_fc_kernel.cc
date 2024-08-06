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

template <typename T, typename Context>
void BatchFCKernel(const Context &dev_ctx,
                   const DenseTensor &input,
                   const DenseTensor &w,
                   const DenseTensor &bias,
                   DenseTensor *out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU,
      true,
      common::errors::Unimplemented("BatchFC only supports GPU now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    batch_fc, CPU, ALL_LAYOUT, phi::BatchFCKernel, float, double) {}
