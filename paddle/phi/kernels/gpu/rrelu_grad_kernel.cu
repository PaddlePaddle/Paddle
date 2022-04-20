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

// #include "paddle/fluid/operators/dropout_impl.cu.h"
#include "paddle/phi/kernels/gpu/rrelu_impl.cu.h"
// #include "paddle/phi/kernels/dropout_grad_kernel.h"
#include "paddle/phi/kernels/rrelu_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RReluGradRawKernel(const Context& dev_ctx,
                        const DenseTensor& mask,
                        const DenseTensor& out_grad,
                        // float p,
                        // bool is_test,
                        // const std::string& mode,
                        DenseTensor* x_grad) {
  x_grad->mutable_data<T>(dev_ctx.GetPlace());
  auto size = x_grad->numel();
  paddle::operators::RReluGradGPUKernelDriver<T>(
      dev_ctx, out_grad, mask, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(rrelu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RReluGradRawKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
