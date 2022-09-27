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

#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/onednn/reduce_kernel_impl.h"

namespace phi {
template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  ReduceKernel<T, Context>(dev_ctx,
                           x,
                           dims,
                           keep_dim,
                           reduce_all,
                           out,
                           dnnl::algorithm::reduction_max);
}
}  // namespace phi

PD_REGISTER_KERNEL(max_raw,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::MaxRawKernel,
                   float,
                   phi::dtype::bfloat16) {}
