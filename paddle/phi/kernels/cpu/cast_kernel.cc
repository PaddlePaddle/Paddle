//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/cpu/cast_impl.h"

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
    PD_VISIT_ALL_CPU_TYPES(out_dtype, "CastInplaceKernelImpl", ([&] {
                             CastInplaceKernelImpl<T, data_t>(
                                 dev_ctx, x, out_dtype, out);
                           }));
  } else {
    PD_VISIT_ALL_CPU_TYPES(out_dtype, "CastKernelImpl", ([&] {
                             CastKernelImpl<T, data_t>(
                                 dev_ctx, x, out_dtype, out);
                           }));
  }
}
#ifdef _WIN32
INSTANTIATE_CAST_KERNEL(float, CPUContext)
INSTANTIATE_CAST_KERNEL(double, CPUContext)
INSTANTIATE_CAST_KERNEL(int, CPUContext)
INSTANTIATE_CAST_KERNEL(int64_t, CPUContext)
INSTANTIATE_CAST_KERNEL(uint8_t, CPUContext)
INSTANTIATE_CAST_KERNEL(bool, CPUContext)
INSTANTIATE_CAST_KERNEL(int16_t, CPUContext)
INSTANTIATE_CAST_KERNEL(phi::float16, CPUContext)
INSTANTIATE_CAST_KERNEL(phi::bfloat16, CPUContext)
#endif
}  // namespace phi

PD_REGISTER_KERNEL(cast,
                   CPU,
                   ALL_LAYOUT,
                   phi::CastKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   int16_t,
                   bool,
                   int8_t,
                   uint8_t,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
