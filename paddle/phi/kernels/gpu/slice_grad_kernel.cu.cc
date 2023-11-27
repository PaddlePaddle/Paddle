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

#include "paddle/phi/kernels/slice_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/slice_grad_kernel_impl.h"

PD_REGISTER_KERNEL(slice_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SliceGradKernel,
                   bool,
                   int,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   int8_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(slice_array_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SliceArrayGradKernel,
                   bool,
                   int,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   int8_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(slice_array_dense_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SliceArrayDenseGradKernel,
                   bool,
                   int,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   int8_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
