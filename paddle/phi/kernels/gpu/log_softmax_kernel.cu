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

#include "paddle/phi/kernels/log_softmax_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {

template <typename T, typename Context>
void LogSoftmaxKernel(const Context &dev_ctx,
                      const DenseTensor &x,
                      int axis,
                      DenseTensor *out) {
  dev_ctx.template Alloc<T>(out);
  phi::SoftmaxForwardCUDAKernelDriver<T, true>(dev_ctx, x, axis, out);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(log_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogSoftmaxKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(log_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogSoftmaxKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
