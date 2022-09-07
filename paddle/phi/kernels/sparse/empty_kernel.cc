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

#include "paddle/phi/kernels/sparse/empty_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void EmptyLikeCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        SparseCooTensor* out) {
  out->set_dims(x.dims());
  *(out->mutable_indices()) = x.indices();

  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_values = out->mutable_non_zero_elements();
  out_values->Resize(x_values.dims());
  dev_ctx.template Alloc<T>(out_values);
}

template <typename T, typename Context>
void EmptyLikeCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        SparseCsrTensor* out) {
  out->set_dims(x.dims());
  *(out->mutable_crows()) = x.crows();
  *(out->mutable_cols()) = x.cols();

  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_values = out->mutable_non_zero_elements();
  out_values->Resize(x_values.dims());
  dev_ctx.template Alloc<T>(out_values);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(empty_like_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::EmptyLikeCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(empty_like_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::EmptyLikeCsrKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(empty_like_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::EmptyLikeCooKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(empty_like_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::EmptyLikeCsrKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
#endif
