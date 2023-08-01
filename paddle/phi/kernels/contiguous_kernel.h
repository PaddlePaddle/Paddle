/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

/**
 * @brief Computes a contiguous in memory tensor containing the same data as
 * input tensor.
 * @param  ctx     device context
 * @param  input   Source tensor need be computed
 * @param  out     The contiguous in memory tensor
 */
template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out);

template <typename T, typename Context>
DenseTensor Contiguous(const Context& dev_ctx, const DenseTensor& input) {
  DenseTensor dense_out;
  MetaTensor meta_input(input);
  MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);
  ContiguousKernel<T, Context>(dev_ctx, input, &dense_out);
  return dense_out;
}
}  // namespace phi
