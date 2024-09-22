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

#include "paddle/phi/kernels/numel_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/numel_kernel_impl.h"

PD_REGISTER_KERNEL(numel,
                   CPU,
                   ALL_LAYOUT,
                   phi::NumelKernel,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(numel,
                   Custom,
                   ALL_LAYOUT,
                   phi::NumelKernel,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
#endif
