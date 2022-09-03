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

#include "paddle/phi/kernels/dropout_kernel.h"

#include "paddle/fluid/operators/dropout_impl.cu.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& seed_tensor,
                      const Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      DenseTensor* out,
                      DenseTensor* mask) {
  bool upscale_in_train = (mode == "upscale_in_train");
  dev_ctx.template Alloc<T>(out);
  if (mask) {
    dev_ctx.template Alloc<uint8_t>(mask);
  }
  paddle::operators::DropoutFwGPUKernelDriver<T>(dev_ctx,
                                                 is_test,
                                                 p.to<float>(),
                                                 upscale_in_train,
                                                 fix_seed,
                                                 seed,
                                                 x,
                                                 seed_tensor.get_ptr(),
                                                 mask,
                                                 out,
                                                 false);
}

template <typename T, typename Context>
void DropoutNdKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& seed_tensor,
                     const Scalar& p,
                     bool is_test,
                     const std::string& mode,
                     int seed,
                     bool fix_seed,
                     const std::vector<int>& axis,
                     DenseTensor* out,
                     DenseTensor* mask) {
  bool upscale_in_train = (mode == "upscale_in_train");
  dev_ctx.template Alloc<T>(out);
  if (mask) {
    dev_ctx.template Alloc<uint8_t>(mask);
  }
  paddle::operators::DropoutFwGPUKernelDriver<T>(dev_ctx,
                                                 is_test,
                                                 p.to<float>(),
                                                 upscale_in_train,
                                                 fix_seed,
                                                 seed,
                                                 x,
                                                 seed_tensor.get_ptr(),
                                                 mask,
                                                 out,
                                                 true);
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout,
                   GPU,
                   ALL_LAYOUT,
                   phi::DropoutRawKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(dropout_nd,
                   GPU,
                   ALL_LAYOUT,
                   phi::DropoutNdKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
}
