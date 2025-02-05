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

#include "paddle/phi/kernels/selected_rows/elementwise_multiply_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"

namespace phi::sr {

template <typename T, typename Context>
void MultiplyRawKernel(const Context& dev_ctx,
                       const SelectedRows& x,
                       const DenseTensor& y,
                       int axis,
                       SelectedRows* out) {
  PADDLE_ENFORCE_EQ(common::product(y.dims()),
                    1,
                    common::errors::InvalidArgument(
                        "For MultiplyKernel, if X is Sparse, Y must "
                        "contain only one element."));
  out->set_rows(x.rows());
  out->set_height(x.height());
  auto z = out->mutable_value();
  z->Resize(x.value().dims());
  dev_ctx.Alloc(z, x.value().dtype());
  MultiplyRawKernel<T, Context>(dev_ctx, x.value(), y, axis, z);
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const SelectedRows& x,
                    const DenseTensor& y,
                    SelectedRows* out) {
  int axis = -1;
  MultiplyRawKernel<T, Context>(dev_ctx, x, y, axis, out);
}

}  // namespace phi::sr

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(multiply_raw_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::MultiplyRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(multiply_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::MultiplyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(multiply_raw_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::MultiplyRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(multiply_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::MultiplyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   complex64,
                   complex128) {}
#endif
