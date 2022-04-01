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

#include "paddle/phi/kernels/selected_rows/assign_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/assign_kernel.h"

namespace phi {
namespace sr {

// Note: use `const paddle::optional<const SelectedRows&> x`
// as input if needed
template <typename Context>
void AssignKernel(const Context& dev_ctx,
                  const SelectedRows& x,
                  SelectedRows* out) {
  out->set_rows(x.rows());
  out->set_height(x.height());
  phi::AssignKernel<Context>(dev_ctx, x.value(), out->mutable_value());
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(assign_sr,
                           CPU,
                           ALL_LAYOUT,
                           phi::sr::AssignKernel<phi::CPUContext>,
                           ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(assign_sr,
                           GPU,
                           ALL_LAYOUT,
                           phi::sr::AssignKernel<phi::GPUContext>,
                           ALL_DTYPE) {}
#endif
