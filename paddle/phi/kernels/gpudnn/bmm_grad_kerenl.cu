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
#ifdef PADDLE_WITH_MUSA
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/bmm_grad_kernel.h"
#include "paddle/phi/kernels/gpudnn/matmul_gpudnn.h"

namespace phi {

template <typename T, typename Context>
void BmmGradGPUDNNKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& out_grad,
                         DenseTensor* x_grad,
                         DenseTensor* y_grad) {
  phi::BmmGPUDNNKernelImpl<T, Context>(
      dev_ctx, out_grad, false, y, true, x_grad);
  phi::BmmGPUDNNKernelImpl<T, Context>(
      dev_ctx, x, true, out_grad, false, y_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(bmm_grad,  // musa_only
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::BmmGradGPUDNNKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#endif
