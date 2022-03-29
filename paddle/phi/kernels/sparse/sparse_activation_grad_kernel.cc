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

#include "paddle/phi/kernels/sparse/sparse_activation_grad_kernel.h"
#include "paddle/phi/kernels/activation_grad_kernel.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SparseReluGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          const SparseCooTensor& out_grad,
                          SparseCooTensor* x_grad) {
  DenseTensor non_zero_indices =
      phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_indices());
  DenseTensor non_zero_elements =
      phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_elements());
  phi::Copy(dev_ctx,
            x.non_zero_indices(),
            dev_ctx.GetPlace(),
            false,
            &non_zero_indices);
  phi::ReluGradKernel<T, Context>(dev_ctx,
                                  x.non_zero_elements(),
                                  out_grad.non_zero_elements(),
                                  &non_zero_elements);
  x_grad->SetMember(non_zero_indices, non_zero_elements, x.dims(), true);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_relu_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseReluGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(sparse_relu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseReluGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
#endif
