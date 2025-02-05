// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/reduce_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& dims,
               DataType out_dtype,
               bool keep_dim,
               DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  SumRawKernel<T, Context>(
      dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(sum,
                   CPU,
                   ALL_LAYOUT,
                   phi::SumKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   uint8_t,
                   int8_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(sum,
                   GPU,
                   ALL_LAYOUT,
                   phi::SumKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   uint8_t,
                   int8_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
#endif

#if defined(PADDLE_WITH_XPU_KP) && !defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(sum, KPS, ALL_LAYOUT, phi::SumKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
#endif

#if defined(PADDLE_WITH_DNNL)
PD_REGISTER_KERNEL(
    sum, OneDNN, ONEDNN, phi::SumKernel, float, phi::dtype::bfloat16) {
  kernel->check_if_onednn_kernel_support_ = phi::ReduceCheckIfOneDNNSupport;
}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(sum,
                   XPU,
                   ALL_LAYOUT,
                   phi::SumKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int8_t,
                   int,
                   int64_t,
                   bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
#endif
