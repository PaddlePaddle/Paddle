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

#include "paddle/phi/kernels/assign_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename Context>
void AssignKernel(const Context& dev_ctx,
                  paddle::optional<const DenseTensor&> x,
                  DenseTensor* out) {
  if (!x.is_initialized()) {
    return;
  }
  auto& x_tensor = *x.get_ptr();
  Copy<Context>(dev_ctx, x_tensor, x_tensor.place(), false, out);
}

// Note: use `const paddle::optional<std::vector<const DenseTensor*>&> x`
// as input if needed
template <typename Context>
void AssignArrayKernel(const Context& dev_ctx,
                       const std::vector<const DenseTensor*>& x,
                       std::vector<DenseTensor*> out) {
  for (size_t i = 0; i < x.size(); ++i) {
    AssignKernel<Context>(dev_ctx, *x[i], out.at(i));
  }
}

}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(
    assign, CPU, ALL_LAYOUT, phi::AssignKernel<phi::CPUContext>, ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(assign_array,
                           CPU,
                           ALL_LAYOUT,
                           phi::AssignArrayKernel<phi::CPUContext>,
                           ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(
    assign, GPU, ALL_LAYOUT, phi::AssignKernel<phi::GPUContext>, ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(assign_array,
                           GPU,
                           ALL_LAYOUT,
                           phi::AssignArrayKernel<phi::GPUContext>,
                           ALL_DTYPE) {}
#endif
