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

#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/sparse/convolution.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const DenseTensor& kernel,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const int groups,
                  const bool subm,
                  SparseCooTensor* out,
                  DenseTensor* rulebook);

template <typename T, typename Context>
SparseCooTensor Conv3d(const Context& dev_ctx,
                       const SparseCooTensor& x,
                       const DenseTensor kernel,
                       const std::vector<int>& paddings,
                       const std::vector<int>& dilations,
                       const std::vector<int>& strides,
                       const int groups,
                       const bool subm,
                       DenseTensor* rulebook) {
  DenseTensor indices = phi::Empty<Context>(
      dev_ctx, DenseTensorMeta(DataType::INT32, {1}, DataLayout::NCHW));
  DenseTensor values =
      phi::Empty<Context>(dev_ctx, DenseTensorMeta(x.dtype(), {1}, x.layout()));
  SparseCooTensor coo(indices, values, x.dims());
  Conv3dKernel<T, Context>(dev_ctx,
                           x,
                           kernel,
                           paddings,
                           dilations,
                           strides,
                           groups,
                           subm,
                           &coo,
                           rulebook);
  return coo;
}

}  // namespace sparse
}  // namespace phi
