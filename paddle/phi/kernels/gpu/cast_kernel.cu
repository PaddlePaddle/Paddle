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

#include "paddle/phi/kernels/cast_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/gpu/cast_impl.h"
namespace phi {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out) {
  if (x.dtype() == out_dtype) {
    if (x.dims() == phi::make_ddim({-1})) {
      *out = x;
      return;
    }
    if (!out->IsSharedWith(x)) {
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }

  if (out->IsSharedWith(x)) {
    auto x_origin = x;
    CastCUDAKernel<T>(dev_ctx, x_origin, out_dtype, out);
  } else {
    CastCUDAKernel<T>(dev_ctx, x, out_dtype, out);
  }
}
#ifdef _WIN32
INSTANTIATE_CAST_KERNEL(float, GPUContext)
INSTANTIATE_CAST_KERNEL(double, GPUContext)
INSTANTIATE_CAST_KERNEL(int, GPUContext)
INSTANTIATE_CAST_KERNEL(int64_t, GPUContext)
INSTANTIATE_CAST_KERNEL(uint8_t, GPUContext)
INSTANTIATE_CAST_KERNEL(bool, GPUContext)
INSTANTIATE_CAST_KERNEL(int16_t, GPUContext)
INSTANTIATE_CAST_KERNEL(phi::float16, GPUContext)
INSTANTIATE_CAST_KERNEL(phi::bfloat16, GPUContext)
#endif
}  // namespace phi

#define PTEN_REGISTER_CAST_CUDA_BASE_TYPE(op_name, ...)        \
  PD_REGISTER_KERNEL(cast,                                     \
                     GPU,                                      \
                     ALL_LAYOUT,                               \
                     phi::CastKernel,                          \
                     float,                                    \
                     double,                                   \
                     int,                                      \
                     int64_t,                                  \
                     int16_t,                                  \
                     bool,                                     \
                     int8_t,                                   \
                     uint8_t,                                  \
                     phi::float16,                             \
                     phi::complex64,                           \
                     phi::complex128,                          \
                     ##__VA_ARGS__) {                          \
    kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED); \
  }

PTEN_REGISTER_CAST_CUDA_BASE_TYPE(cast,
                                  phi::bfloat16,
                                  phi::dtype::float8_e4m3fn,
                                  phi::dtype::float8_e5m2)
