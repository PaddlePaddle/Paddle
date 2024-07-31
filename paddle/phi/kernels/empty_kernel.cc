/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/common/macros.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void EmptyKernel(const Context& dev_ctx,
                 const IntArray& shape,
                 DataType dtype UNUSED,
                 DenseTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);
}

template <typename T, typename Context>
void EmptyLikeKernel(const Context& dev_ctx,
                     const DenseTensor& x UNUSED,
                     DataType dtype UNUSED,
                     DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
}

}  // namespace phi

PD_REGISTER_KERNEL(empty,
                   CPU,
                   ALL_LAYOUT,
                   phi::EmptyKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(empty_like,
                   CPU,
                   ALL_LAYOUT,
                   phi::EmptyLikeKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(empty,
                   GPU,
                   ALL_LAYOUT,
                   phi::EmptyKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(empty_like,
                   GPU,
                   ALL_LAYOUT,
                   phi::EmptyLikeKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(empty,
                   XPU,
                   ALL_LAYOUT,
                   phi::EmptyKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(empty_like,
                   XPU,
                   ALL_LAYOUT,
                   phi::EmptyLikeKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(empty,
                   Custom,
                   ALL_LAYOUT,
                   phi::EmptyKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(empty_like,
                   Custom,
                   ALL_LAYOUT,
                   phi::EmptyLikeKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
#endif
