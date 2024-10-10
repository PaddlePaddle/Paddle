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
#include "paddle/phi/kernels/bmm_kernel.h"
#include "paddle/phi/kernels/gpudnn/matmul_gpudnn.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void BmmGPUDNNKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  VLOG(3) << "Call BmmGPUDNNKernel";
  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0 || y.numel() == 0) {
    return;
  }
  phi::BmmGPUDNNKernelImpl<T>(dev_ctx, x, false, y, false, out);
}

}  // namespace phi

// will fail to call this GPUDNN kernel in static graph mode
// bmm op has no "use_cudnn" attribute, which causes dispatching
// to GPUDNN kernel to fail, see paddle/fluid/framework/operator.cc#L1565
// though we can add this attribute when creating bmm OP, it might not be a good
// choice
PD_REGISTER_KERNEL(bmm,  // musa_only
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::BmmGPUDNNKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
