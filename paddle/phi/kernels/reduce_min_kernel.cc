// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/reduce_min_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/reduce_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& dims,
               bool keep_dim,
               DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  PADDLE_ENFORCE_GT(
      x.numel(),
      0,
      errors::InvalidArgument("Zero-size tensor to reduction operation minimum "
                              "which has no identity."));
  MinRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    min, CPU, ALL_LAYOUT, phi::MinKernel, float, double, int, int64_t) {}

#if defined(PADDLE_WITH_CUDA)
PD_REGISTER_KERNEL(min,
                   GPU,
                   ALL_LAYOUT,
                   phi::MinKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

#if defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(
    min, GPU, ALL_LAYOUT, phi::MinKernel, float, double, int, int64_t) {}
#endif

#if defined(PADDLE_WITH_XPU_KP) && !defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(min, KPS, ALL_LAYOUT, phi::MinKernel, float) {}
#endif

#if defined(PADDLE_WITH_DNNL)
PD_REGISTER_KERNEL(
    min, OneDNN, ONEDNN, phi::MinKernel, float, phi::dtype::bfloat16) {
  kernel->check_if_onednn_kernel_support_ = phi::ReduceCheckIfOneDNNSupport;
}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(min,
                   XPU,
                   ALL_LAYOUT,
                   phi::MinKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
