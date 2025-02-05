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

#include "paddle/phi/kernels/selected_rows/shape_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/shape_kernel.h"

namespace phi::sr {

template <typename T, typename Context>
void ShapeKernel(const Context& ctx,
                 const SelectedRows& input,
                 DenseTensor* out) {
  phi::ShapeKernel<T, Context>(ctx, input.value(), out);
}

template <typename T, typename Context>
void Shape64Kernel(const Context& ctx,
                   const SelectedRows& input,
                   DenseTensor* out) {
  phi::Shape64Kernel<T, Context>(ctx, input.value(), out);
}

}  // namespace phi::sr

PD_REGISTER_KERNEL(shape_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::ShapeKernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(shape_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::ShapeKernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(shape_sr,
                   XPU,
                   ALL_LAYOUT,
                   phi::sr::ShapeKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(shape_sr,
                   Custom,
                   ALL_LAYOUT,
                   phi::sr::ShapeKernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}
#endif

PD_REGISTER_KERNEL(shape64_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::Shape64Kernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(shape64_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::Shape64Kernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
#endif

#if defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(shape64_sr,
                   XPU,
                   ALL_LAYOUT,
                   phi::sr::Shape64Kernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(shape64_sr,
                   Custom,
                   ALL_LAYOUT,
                   phi::sr::Shape64Kernel,
                   bool,
                   int,
                   int8_t,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
#endif
