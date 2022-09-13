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

#include "paddle/phi/kernels/sparse/elementwise_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void ElementWiseAddCooGradKernel(const Context& dev_ctx,
                                 const SparseCooTensor& x,
                                 const SparseCooTensor& y,
                                 const SparseCooTensor& dout,
                                 SparseCooTensor* dx,
                                 SparseCooTensor* dy) {
  if (dx) {
    EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  }

  if (dy) {
    EmptyLikeCooKernel<T, Context>(dev_ctx, y, dy);
    Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(add_coo_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
