// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/complex_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/complex_kernel_impl.h"

PD_REGISTER_KERNEL(conj,
                   GPU,
                   ALL_LAYOUT,
                   phi::ConjKernel,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(
    real, GPU, ALL_LAYOUT, phi::RealKernel, phi::complex64, phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(
    imag, GPU, ALL_LAYOUT, phi::ImagKernel, phi::complex64, phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(
    complex, GPU, ALL_LAYOUT, phi::ComplexKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
