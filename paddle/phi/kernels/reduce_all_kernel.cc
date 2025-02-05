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

#include "paddle/phi/kernels/reduce_all_kernel.h"
#include "glog/logging.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

namespace phi {

template <typename T, typename Context>
void AllKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out) {
  auto x_dim = x.dims();
  for (int i = 0; i < x_dim.size(); i++) {
    PADDLE_ENFORCE_LE(
        0,
        x_dim[i],
        errors::InvalidArgument(
            "The dims of Input(X) should be greater than or equal to 0."));
  }
  if (x.numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    if (out->numel() > 0) {
      std::vector<int64_t> vec_dims = common::vectorize(out->dims());
      phi::Full<bool, Context>(dev_ctx, phi::IntArray(vec_dims), 1, out);
    }
    return;
  }
  bool reduce_all = recompute_reduce_all(x, dims);
  AllRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(all,
                   CPU,
                   ALL_LAYOUT,
                   phi::AllKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(all,
                   GPU,
                   ALL_LAYOUT,
                   phi::AllKernel,
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

#if defined(PADDLE_WITH_XPU_KP)
PD_REGISTER_KERNEL(all, KPS, ALL_LAYOUT, phi::AllKernel, bool) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(all, XPU, ALL_LAYOUT, phi::AllKernel, bool) {}
#endif
