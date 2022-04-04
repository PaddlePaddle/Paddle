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

#include "paddle/phi/kernels/sparse/activation_kernel.h"

#include "paddle/phi/kernels/sparse/utils.h"

DEFINE_AND_REGISTER_SPARSE_UNARY_KERNEL(relu, ReluKernel)
DEFINE_AND_REGISTER_SPARSE_UNARY_KERNEL(sqrt, SqrtKernel)

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SparseCooXX(const Context& dev_ctx,
                 const SparseCooTensor& x,
                 SparseCooTensor* out) {
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(
    sparse_coo_xx, CPU, ALL_LAYOUT, phi::sparse::SparseCooXX, float, double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
