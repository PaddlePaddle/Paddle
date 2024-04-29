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
 * @brief copy memory with stride.
 * @param  ctx     device context
 * @param  input   Source tensor need copy from
 * @param  out_stride   Dist tensor's stride
 * @param  out     Dist tensor need copy to
 */
template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out);

template <typename T, typename Context>
DenseTensor StridedCopy(const Context& dev_ctx,
                        const DenseTensor& input,
                        const std::vector<int64_t>& dims,
                        const std::vector<int64_t>& out_stride,
                        size_t offset) {
  DenseTensor dense_out;
  StridedCopyKernel<T, Context>(
      dev_ctx, input, dims, out_stride, offset, &dense_out);
  return dense_out;
}
}  // namespace phi
