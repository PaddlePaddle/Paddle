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

#include "paddle/phi/kernels/log_softmax_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context &dev_ctx,
                          const DenseTensor &out,
                          const DenseTensor &out_grad,
                          int axis,
                          DenseTensor *x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  const int rank = out.dims().size();
  // For 0D Tensor
  if (rank == 0) {
    phi::funcs::set_constant(dev_ctx, x_grad, static_cast<T>(0.0));
    return;
  }
  phi::SoftmaxBackwardCUDAKernelDriver<T, true>(
      dev_ctx, out, out_grad, axis, x_grad);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(log_softmax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogSoftmaxGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#else
PD_REGISTER_KERNEL(log_softmax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogSoftmaxGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
