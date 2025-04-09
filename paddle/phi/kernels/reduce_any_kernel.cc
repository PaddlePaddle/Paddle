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

#include "paddle/phi/kernels/reduce_any_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename Context>
void AnyKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    if (out->numel() > 0) {
      std::vector<int64_t> vec_dims = common::vectorize(out->dims());
      phi::Full<bool, Context>(dev_ctx, phi::IntArray(vec_dims), 0, out);
    }
    return;
  }
  bool reduce_all = recompute_reduce_all(x, dims);
  AnyRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(any,
                   CPU,
                   ALL_LAYOUT,
                   phi::AnyKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   bool,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(any,
                   GPU,
                   ALL_LAYOUT,
                   phi::AnyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
#endif

#if defined(PADDLE_WITH_XPU_KP) && !defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(any, KPS, ALL_LAYOUT, phi::AnyKernel, bool) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(any, XPU, ALL_LAYOUT, phi::AnyKernel, bool) {}
#endif
