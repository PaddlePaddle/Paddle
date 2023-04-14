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
#include "paddle/phi/kernels/sparse/mask_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void ValuesCooGradKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         const DenseTensor& out_grad,
                         SparseCooTensor* x_grad);

template <typename T, typename Context>
void CooToDenseGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          const DenseTensor& out_grad,
                          SparseCooTensor* x_grad);

template <typename T, typename Context>
void SparseCooTensorGradKernel(const Context& dev_ctx,
                               const DenseTensor& indices,
                               const SparseCooTensor& out_grad,
                               DenseTensor* values_grad) {
  MaskHelperCooKernel<T, Context>(dev_ctx, out_grad, indices, values_grad);
}

}  // namespace sparse
}  // namespace phi
