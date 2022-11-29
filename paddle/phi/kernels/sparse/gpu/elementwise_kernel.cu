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

#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/sparse/elementwise_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
void ElementWiseAddCooGPUKernel(const GPUContext& dev_ctx,
                                const SparseCooTensor& x,
                                const SparseCooTensor& y,
                                SparseCooTensor* out) {
  // TODO(zhangkaiuo): to support universal sparse + sparse
  const auto& x_indices = x.indices();
  const auto& y_indices = y.indices();
  PADDLE_ENFORCE_EQ(
      x_indices.numel(),
      y_indices.numel(),
      phi::errors::PreconditionNotMet(
          "The numel of x.indices() and y.indices() should be equal"));
  const IntT* x_indices_ptr = x_indices.data<IntT>();
  const IntT* y_indices_ptr = y_indices.data<IntT>();
#ifdef PADDLE_WITH_HIP
  bool is_same = thrust::equal(thrust::hip::par.on(dev_ctx.stream()),
#else
  bool is_same = thrust::equal(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                               x_indices_ptr,
                               x_indices_ptr + x_indices.numel(),
                               y_indices_ptr);
  PADDLE_ENFORCE_EQ(
      is_same,
      true,
      phi::errors::PreconditionNotMet(
          "Currently, ElementWiseAddCooKernel only supports the case "
          "where x and y have the same indices"));
  EmptyLikeCooKernel<T, GPUContext>(dev_ctx, x, out);
  phi::AddKernel<T, GPUContext>(
      dev_ctx, x.values(), y.values(), out->mutable_values());
  out->SetIndicesDict(x.GetIndicesDict());
}

template <typename T, typename Context>
void ElementWiseAddCooKernel(const Context& dev_ctx,
                             const SparseCooTensor& x,
                             const SparseCooTensor& y,
                             SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "VerifyIndices", ([&] {
                                 ElementWiseAddCooGPUKernel<T, data_t>(
                                     dev_ctx, x, y, out);
                               }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(add_coo_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCooKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(add_coo_dense,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddDenseKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
