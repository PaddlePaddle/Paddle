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

#include "paddle/phi/kernels/sparse/elementwise_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {
/*
 * out.values() = x.values() + y.values()
 */
template <typename T, typename Context>
void ValuesAddCooCooKernel(const Context& dev_ctx,
                           const SparseCooTensor& x,
                           const SparseCooTensor& y,
                           SparseCooTensor* out) {
  // TODO(zkh2016): assert(x.indices() == y.indices())
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);
  phi::AddKernel<T, Context>(dev_ctx,
                             x.non_zero_elements(),
                             y.non_zero_elements(),
                             out->mutable_non_zero_elements());
}

/*
 * out.values() = x.values() + values
 */
template <typename T, typename Context>
void ValuesAddCooDenseKernel(const Context& dev_ctx,
                             const SparseCooTensor& x,
                             const DenseTensor& y,
                             SparseCooTensor* out) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);
  phi::AddKernel<T, Context>(
      dev_ctx, x.non_zero_elements(), y, out->mutable_non_zero_elements());
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(values_add_coo_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ValuesAddCooCooKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(values_add_coo_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ValuesAddCooDenseKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
