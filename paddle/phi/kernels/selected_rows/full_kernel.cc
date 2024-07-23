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

#include "paddle/phi/kernels/selected_rows/full_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi::sr {

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DataType dtype,
                SelectedRows* out) {
  phi::FullKernel<T>(dev_ctx, shape, val, dtype, out->mutable_value());
}

template <typename T, typename Context>
void FullWithTensorKernel(const Context& dev_ctx,
                          const DenseTensor& value,
                          const IntArray& shape,
                          DataType dtype,
                          SelectedRows* out) {
  phi::FullWithTensorKernel<T>(
      dev_ctx, value, shape, dtype, out->mutable_value());
}

}  // namespace phi::sr

PD_REGISTER_KERNEL(full_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::FullKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(full_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::FullKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(full_sr,
                   XPU,
                   ALL_LAYOUT,
                   phi::sr::FullKernel,
                   float,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16) {}
#endif

PD_REGISTER_KERNEL(full_with_tensor_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::FullWithTensorKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(full_with_tensor_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::FullWithTensorKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);
}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(full_with_tensor_sr,
                   XPU,
                   ALL_LAYOUT,
                   phi::sr::FullWithTensorKernel,
                   float,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);
}
#endif
