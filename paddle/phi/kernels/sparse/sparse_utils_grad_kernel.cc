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

#include "paddle/phi/kernels/sparse/sparse_utils_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/sparse_mask_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void CooValuesGradKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const DenseTensor& out_grad,
                         SparseCooTensor* x_grad) {
  x_grad->SetMember(x.non_zero_indices(), out_grad, x.dims(), true);
}

template <typename T, typename Context>
void SparseCooToDenseGradKernel(const Context& dev_ctx,
                                const SparseCooTensor& x,
                                const DenseTensor& out_grad,
                                SparseCooTensor* x_grad) {
  SparseMaskKernel<T, Context>(dev_ctx, out_grad, x, x_grad);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(coo_values_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CooValuesGradKernel,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_coo_to_dense_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooToDenseGradKernel,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_coo_tensor_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooTensorGradKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(coo_values_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CooValuesGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
PD_REGISTER_KERNEL(sparse_coo_to_dense_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooToDenseGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
PD_REGISTER_KERNEL(sparse_coo_tensor_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooTensorGradKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
#endif
