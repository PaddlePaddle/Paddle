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

#include "paddle/phi/kernels/sparse/full_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context>
void FullValue(const Context& dev_ctx, DenseTensor* tensor, T val) {
  dev_ctx.template Alloc<T>(tensor);
  auto t = phi::EigenVector<T>::Flatten(*tensor);
  t.device(*dev_ctx.eigen_device()) = t.constant(val);
}

template <typename T, typename Context>
void CooFullLikeKernel(const Context& dev_ctx,
                       const SparseCooTensor& x,
                       const Scalar& val,
                       DataType dtype,
                       SparseCooTensor* out) {
  phi::Copy<Context>(dev_ctx,
                     x.non_zero_indices(),
                     dev_ctx.GetPlace(),
                     false,
                     out->mutable_non_zero_indices());

  DenseTensor* values = out->mutable_non_zero_elements();
  values->Resize(x.non_zero_elements().dims());
  dev_ctx.template Alloc<T>(values);
  FullValue<T, Context>(dev_ctx, values, val.to<T>());

  out->set_dims(x.dims());
}

template <typename T, typename Context>
void CsrFullLikeKernel(const Context& dev_ctx,
                       const SparseCsrTensor& x,
                       const Scalar& val,
                       DataType dtype,
                       SparseCsrTensor* out) {
  phi::Copy<Context>(dev_ctx,
                     x.non_zero_crows(),
                     dev_ctx.GetPlace(),
                     false,
                     out->mutable_non_zero_crows());

  phi::Copy<Context>(dev_ctx,
                     x.non_zero_cols(),
                     dev_ctx.GetPlace(),
                     false,
                     out->mutable_non_zero_cols());

  DenseTensor* values = out->mutable_non_zero_elements();
  values->Resize(x.non_zero_elements().dims());
  dev_ctx.template Alloc<T>(values);
  FullValue<T, Context>(dev_ctx, values, val.to<T>());

  out->set_dims(x.dims());
}

}  // namespace phi

PD_REGISTER_KERNEL(coo_full_like,
                   CPU,
                   ALL_LAYOUT,
                   phi::CooFullLikeKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(csr_full_like,
                   CPU,
                   ALL_LAYOUT,
                   phi::CsrFullLikeKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
