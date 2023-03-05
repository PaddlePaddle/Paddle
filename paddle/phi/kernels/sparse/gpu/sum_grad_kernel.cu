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

template <typename T, typename Context>
void SumCooGradKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const SparseCooTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCooTensor* dx) {}

template <typename T, typename Context>
void SumCsrGradKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const SparseCsrTensor& dout,
                      const IntArray& axis,
                      bool keep_dim,
                      SparseCsrTensor* dx) {}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sum_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCooGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(sum_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SumCsrGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
