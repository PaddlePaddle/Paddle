// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"

namespace phi {
namespace sparse {

// just copy from paddle\phi\kernels\sparse\cpu\reshape_grad_kernel.cc
template <typename T, typename Context>
void ReshapeCooGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          const SparseCooTensor& dout,
                          SparseCooTensor* dx) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
  phi::IntArray x_shape(common::vectorize(x.dims()));
  ReshapeCooKernel<T, Context>(dev_ctx, dout, x_shape, dx);
}

// just copy from paddle\phi\kernels\sparse\cpu\reshape_grad_kernel.cc
template <typename T, typename Context>
void ReshapeCsrGradKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          const SparseCsrTensor& dout,
                          SparseCsrTensor* dx) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);
  phi::IntArray x_shape(common::vectorize(x.dims()));
  ReshapeCsrKernel<T, Context>(dev_ctx, dout, x_shape, dx);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(reshape_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCooGradKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(reshape_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCsrGradKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
