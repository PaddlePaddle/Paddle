// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/squeeze_grad_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/reshape_kernel.h"

namespace phi {

template <typename Context>
void SqueezeGradStridedKernel(const Context& dev_ctx,
                              const DenseTensor& xshape,
                              const DenseTensor& dout,
                              const IntArray& axes UNUSED,
                              DenseTensor* dx) {
  auto xshape_dims = xshape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());
  ReshapeStridedKernel<Context>(
      dev_ctx, dout, IntArray(phi::vectorize<int64_t>(x_dims)), dx, nullptr);
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    squeeze_grad, STRIDED, phi::SqueezeGradStridedKernel) {}
