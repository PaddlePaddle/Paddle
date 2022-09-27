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

#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/onednn/reduce_kernel_impl.h"

namespace phi {
template <typename T, typename Context>
void SumGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& out_grad,
                   const IntArray& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* x_grad) {
  ReduceGradKernel<T, Context>(dev_ctx,
                               x,
                               out_grad,
                               dims,
                               keep_dim,
                               reduce_all,
                               x_grad,
                               dnnl::algorithm::binary_add,
                               dnnl::algorithm::reduction_sum,
                               0.0f,
                               1.0f);
}
}  // namespace phi

PD_REGISTER_KERNEL(sum_grad,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::SumGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
