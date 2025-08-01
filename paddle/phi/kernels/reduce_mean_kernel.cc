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

#include "paddle/phi/kernels/reduce_mean_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/reduce_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& dims,
                bool keep_dim,
                DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (std::is_same<T, int>::value || std::is_same<T, int64_t>::value ||
      std::is_same<T, bool>::value) {
    using Type =
        typename std::conditional<std::is_same<T, int>::value ||
                                      std::is_same<T, int64_t>::value ||
                                      std::is_same<T, bool>::value,
                                  float,
                                  T>::type;
    DenseTensor x_float =
        phi::Cast<T, Context>(dev_ctx, x, phi::DataType::FLOAT32);
    DenseTensor* out_float = new DenseTensor();
    out_float->Resize(out->dims());
    MeanRawKernel<Type>(
        dev_ctx, x_float, dims, keep_dim, reduce_all, out_float);
    phi::CastKernel<Type, Context>(dev_ctx, *out_float, x.dtype(), out);
  } else {
    MeanRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(mean,
                   CPU,
                   ALL_LAYOUT,
                   phi::MeanKernel,
                   float,
                   double,
                   bool,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(mean,
                   GPU,
                   ALL_LAYOUT,
                   phi::MeanKernel,
                   float,
                   double,
                   bool,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#if defined(PADDLE_WITH_XPU_KP) && !defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(mean, KPS, ALL_LAYOUT, phi::MeanKernel, float) {}
#endif

#if defined(PADDLE_WITH_DNNL)
PD_REGISTER_KERNEL(
    mean, OneDNN, ONEDNN, phi::MeanKernel, float, phi::dtype::bfloat16) {
  kernel->check_if_onednn_kernel_support_ = phi::ReduceMeanCheckIfOneDNNSupport;
}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(mean,
                   XPU,
                   ALL_LAYOUT,
                   phi::MeanKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
