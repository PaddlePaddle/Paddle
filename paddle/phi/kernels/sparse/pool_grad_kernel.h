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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void MaxPoolCooGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          const DenseTensor& rulebook,
                          const DenseTensor& counter,
                          const SparseCooTensor& out,
                          const SparseCooTensor& out_grad,
                          const std::vector<int>& kernel_sizes,
                          SparseCooTensor* x_grad);

template <typename T, typename Context>
SparseCooTensor MaxPoolCooGrad(const Context& dev_ctx,
                               const SparseCooTensor& x,
                               const DenseTensor& rulebook,
                               const DenseTensor& counter,
                               const SparseCooTensor& out,
                               const SparseCooTensor& out_grad,
                               const std::vector<int>& kernel_sizes) {
  SparseCooTensor x_grad;
  MaxPoolCooGradKernel<T, Context>(
      dev_ctx, x, rulebook, counter, out, out_grad, kernel_sizes, &x_grad);
  return x_grad;
}

}  // namespace sparse
}  // namespace phi
