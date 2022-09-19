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

// std::vector<int> get_gpu_grad_perm(std::vector<int> perm) {
//   std::vector<int> grad_perm(perm.size());
//   for (unsigned int i = 0; i < perm.size(); ++i) {
//     grad_perm[perm[i]] = i;
//   }
//   return grad_perm;
// }


// just copy from paddle\phi\kernels\sparse\cpu\reshape_grad_kernel.cc
template <typename T, typename Context>
void ReshapeCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const SparseCooTensor& dout,
                            // const std::vector<int>& perm,
                            SparseCooTensor* dx) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
  // std::vector<int> grad_perm = get_gpu_grad_perm(perm);
  // TransposeCooKernel<T, Context>(dev_ctx, dout, grad_perm, dx);
  std::vector<int64_t> x_shape(x.dims().size());
  for (int i=0; i<x.dims().size(); ++i) {
    x_shape.push_back(x.dims()[i]);
  }
  ReshapeCooKernel<T, Context>(dev_ctx, dout, x_shape, dx);
}

// template <typename T, typename Context>
// void TransposeCsrGradKernel(const Context& dev_ctx,
//                             const SparseCsrTensor& x,
//                             const SparseCsrTensor& dout,
//                             const std::vector<int>& perm,
//                             SparseCsrTensor* dx) {
//   EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);
//   std::vector<int> grad_perm = get_gpu_grad_perm(perm);
//   TransposeCsrKernel<T, Context>(dev_ctx, dout, grad_perm, dx);
// }

// just copy from paddle\phi\kernels\sparse\cpu\reshape_grad_kernel.cc
template <typename T, typename Context>
void ReshapeCsrGradKernel(const Context& dev_ctx,
                            const SparseCsrTensor& x,
                            const SparseCsrTensor& dout,
                            // const std::vector<int>& perm,
                            SparseCsrTensor* dx) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);
  std::vector<int64_t> x_shape(x.dims().size());
  for (int i=0; i<x.dims().size(); ++i) {
    x_shape.push_back(x.dims()[i]);
  }
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