/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/softmax_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxGPUDNNKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         int axis,
                         DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  const int rank = x.dims().size();
  // For 0D Tensor
  if (rank == 0) {
    phi::funcs::set_constant(dev_ctx, out, static_cast<T>(1.0));
    return;
  }

  SoftmaxForwardCUDAKernelDriver<T>(dev_ctx, x, axis, out);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(softmax,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::SoftmaxGPUDNNKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(softmax,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::SoftmaxGPUDNNKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(softmax,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::SoftmaxGPUDNNKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
#endif
