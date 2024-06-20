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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"

namespace phi::sparse {

std::vector<int> get_cpu_grad_perm(std::vector<int> perm) {
  std::vector<int> grad_perm(perm.size());
  for (unsigned int i = 0; i < perm.size(); ++i) {
    grad_perm[perm[i]] = static_cast<int>(i);
  }
  return grad_perm;
}

template <typename T, typename Context>
void TransposeCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& dout,
                            const std::vector<int>& perm,
                            SparseCooTensor* dx) {
  std::vector<int> grad_perm = get_cpu_grad_perm(perm);
  TransposeCooKernel<T, Context>(dev_ctx, dout, grad_perm, dx);
}

template <typename T, typename Context>
void TransposeCsrGradKernel(const Context& dev_ctx,
                            const SparseCsrTensor& dout,
                            const std::vector<int>& perm,
                            SparseCsrTensor* dx) {
  std::vector<int> grad_perm = get_cpu_grad_perm(perm);
  TransposeCsrKernel<T, Context>(dev_ctx, dout, grad_perm, dx);
}
}  // namespace phi::sparse

PD_REGISTER_KERNEL(transpose_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::TransposeCooGradKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(transpose_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::TransposeCsrGradKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
