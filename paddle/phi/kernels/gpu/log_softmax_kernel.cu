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
<<<<<<< HEAD
#include "paddle/phi/kernels/funcs/math_function.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {

template <typename T, typename Context>
void LogSoftmaxKernel(const Context &dev_ctx,
                      const DenseTensor &x,
                      int axis,
                      DenseTensor *out) {
<<<<<<< HEAD
  const int rank = x.dims().size();

  dev_ctx.template Alloc<T>(out);
  // For 0D Tensor
  if (rank == 0) {
    phi::funcs::set_constant(dev_ctx, out, 0.0);
    return;
  }
=======
  dev_ctx.template Alloc<T>(out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
