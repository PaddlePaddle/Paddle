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

#include "paddle/phi/kernels/isfinite_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/isfinite_kernel_impl.h"

PD_REGISTER_KERNEL(isinf,
                   GPU,
                   ALL_LAYOUT,
                   phi::IsinfKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t,
                   int16_t,
                   int8_t,
                   uint8_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(isnan,
                   GPU,
                   ALL_LAYOUT,
                   phi::IsnanKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(isfinite,
                   GPU,
                   ALL_LAYOUT,
                   phi::IsfiniteKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

#ifdef _WIN32
namespace phi {
INSTANTIATE_ISFINITE_KERNEL_Isnan(float, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isnan(double, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isnan(int, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isnan(int64_t, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isnan(phi::float16, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isnan(phi::bfloat16, GPUContext);

INSTANTIATE_ISFINITE_KERNEL_Isinf(float, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isinf(double, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isinf(int, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isinf(int64_t, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isinf(phi::float16, GPUContext);
INSTANTIATE_ISFINITE_KERNEL_Isinf(phi::bfloat16, GPUContext);
}  // namespace phi
#endif
