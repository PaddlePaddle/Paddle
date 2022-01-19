//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/flatten_grad_kernel.h"
#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/copy_kernel.h"

namespace pten {

template <typename T, typename Context>
void FlattenGradKernel(const Context& dev_ctx,
                       const DenseTensor& out_grad,
                       const DenseTensor& xshape,
                       DenseTensor* x_grad) {
  auto xshape_dims = xshape.dims();
  auto x_dims =
      paddle::framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
  pten::Copy(dev_ctx, out_grad, false, x_grad);
  x_grad->Resize(x_dims);
}

}  // namespace pten

PT_REGISTER_CTX_KERNEL(flatten_grad,
                       CPU,
                       ALL_LAYOUT,
                       pten::FlattenGradKernel,
                       float,
                       double,
                       uint8_t,
                       int8_t,
                       int,
                       int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_REGISTER_CTX_KERNEL(flatten_grad,
                       GPU,
                       ALL_LAYOUT,
                       pten::FlattenGradKernel,
                       float,
                       paddle::platform::float16,
                       double,
                       uint8_t,
                       int8_t,
                       int,
                       int64_t) {}

#endif

#ifdef PADDLE_WITH_XPU
PT_REGISTER_CTX_KERNEL(flatten_grad,
                       XPU,
                       ALL_LAYOUT,
                       pten::FlattenGradKernel,
                       float,
                       paddle::platform::float16,
                       int8_t,
                       int,
                       int64_t) {}

#endif
