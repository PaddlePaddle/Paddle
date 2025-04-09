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

#include "paddle/phi/kernels/full_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FullBatchSizeLikeKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const std::vector<int>& shape UNUSED,
                             const Scalar& val,
                             DataType dtype,
                             int x_batch_size_dim,
                             int out_batch_size_dim,
                             DenseTensor* out) {
  if (!x.lod().empty() && x_batch_size_dim == 0) {
    // set the correct batch size for the DenseTensor.
    auto odims = out->dims();
    odims[out_batch_size_dim] = static_cast<int>(x.lod().back().size()) - 1;
    FullKernel<T, Context>(dev_ctx, common::vectorize(odims), val, dtype, out);
  }
  FullLikeKernel<T, Context>(dev_ctx, x, val, dtype, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(full_batch_size_like,
                   CPU,
                   ALL_LAYOUT,
                   phi::FullBatchSizeLikeKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(full_batch_size_like,
                   GPU,
                   ALL_LAYOUT,
                   phi::FullBatchSizeLikeKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
#endif
