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

#include "paddle/phi/kernels/dropout_grad_kernel.h"

#include "paddle/fluid/operators/dropout_impl.cu.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DropoutGradRawKernel(const Context& dev_ctx,
                          const DenseTensor& mask,
                          const DenseTensor& out_grad,
                          const Scalar& p,
                          bool is_test,
                          const std::string& mode,
                          DenseTensor* x_grad) {
  bool upscale_in_train = (mode == "upscale_in_train");
  dev_ctx.template Alloc<T>(x_grad);
  paddle::operators::DropoutGradGPUKernelDriver<T>(dev_ctx,
                                                   is_test,
                                                   p.to<float>(),
                                                   upscale_in_train,
                                                   out_grad,
                                                   mask,
                                                   x_grad,
                                                   false);
}

template <typename T, typename Context>
void DropoutNdGradKernel(const Context& dev_ctx,
                         const DenseTensor& mask,
                         const DenseTensor& out_grad,
                         const Scalar& p,
                         bool is_test,
                         const std::string& mode,
                         const std::vector<int>& axis,
                         DenseTensor* x_grad) {
  bool upscale_in_train = (mode == "upscale_in_train");
  dev_ctx.template Alloc<T>(x_grad);
  paddle::operators::DropoutGradGPUKernelDriver<T>(dev_ctx,
                                                   is_test,
                                                   p.to<float>(),
                                                   upscale_in_train,
                                                   out_grad,
                                                   mask,
                                                   x_grad,
                                                   true);
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DropoutGradRawKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(dropout_nd_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DropoutNdGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
