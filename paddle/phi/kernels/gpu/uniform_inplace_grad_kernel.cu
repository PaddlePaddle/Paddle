/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/uniform_inplace_grad_kernel.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename Context>
void UniformInplaceGradKernel(const Context& ctx,
                              const DenseTensor& out_grad,
                              float min,
                              float max,
                              int seed,
                              int diag_num,
                              int diag_step,
                              float diag_val,
                              DenseTensor* x_grad) {
  auto dims = common::vectorize(x_grad->dims());
  float value = static_cast<float>(0.0f);
  phi::FullKernel<T>(ctx, dims, value, phi::DataType::UNDEFINED, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_inplace_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::UniformInplaceGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
