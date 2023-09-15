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

#include "paddle/phi/kernels/matmul_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/matmul_kernel_impl.h"

#ifdef PADDLE_WITH_CUDA
PD_REGISTER_KERNEL(matmul,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   int8_t) {
  if (kernel_key.dtype() == phi::DataType::INT8) {
    kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
  }
}
#else
PD_REGISTER_KERNEL(matmul,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  if (kernel_key.dtype() == phi::DataType::INT8) {
    kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
  }
}
#endif

#ifdef PADDLE_WITH_CUDA
PD_REGISTER_KERNEL(matmul_with_flatten,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenKernel,
                   int8_t,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::INT8) {
    kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
  }
}
#else
PD_REGISTER_KERNEL(matmul_with_flatten,
                   GPU,
                   ALL_LAYOUT,
                   phi::MatmulWithFlattenKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::INT8) {
    kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
  }
}
#endif
