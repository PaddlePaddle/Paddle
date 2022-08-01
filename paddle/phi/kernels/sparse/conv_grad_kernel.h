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
void Conv3dCooGradKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const DenseTensor& kernel,
                         const SparseCooTensor& out,
                         const DenseTensor& rulebook,
                         const DenseTensor& counter,
                         const SparseCooTensor& out_grad,
                         const std::vector<int>& paddings,
                         const std::vector<int>& dilations,
                         const std::vector<int>& strides,
                         const int groups,
                         const bool subm,
                         const std::string& key,
                         SparseCooTensor* x_grad,
                         DenseTensor* kernel_grad);

template <typename T, typename Context>
std::tuple<SparseCooTensor, DenseTensor> Conv3dCooGrad(
    const Context& dev_ctx,
    const SparseCooTensor& x,
    const DenseTensor& kernel,
    const SparseCooTensor& out,
    const DenseTensor& rulebook,
    const DenseTensor& counter,
    const SparseCooTensor& out_grad,
    const std::vector<int>& paddings,
    const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const int groups,
    const bool subm,
    const std::string& key) {
  SparseCooTensor x_grad;
  DenseTensor kernel_grad;

  // TODO(zhangkaihuo): call InferMeta func here
  Conv3dCooGradKernel<T, Context>(dev_ctx,
                                  x,
                                  kernel,
                                  out,
                                  rulebook,
                                  counter,
                                  out_grad,
                                  paddings,
                                  dilations,
                                  strides,
                                  groups,
                                  subm,
                                  key,
                                  &x_grad,
                                  &kernel_grad);
  return std::make_tuple(x_grad, kernel_grad);
}

}  // namespace sparse
}  // namespace phi
