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

#include "paddle/phi/kernels/abs_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"

namespace phi {

template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  int r = xpu::abs<XPUType>(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(x.data<T>()),
                            reinterpret_cast<XPUType*>(out->data<T>()),
                            x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "abs");
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void AbsKernel<phi::complex64, XPUContext>(const XPUContext& dev_ctx,
                                           const DenseTensor& x,
                                           DenseTensor* out) {
  using T = phi::complex64;
  using RealT = phi::dtype::Real<T>;

  if (x.numel() == 0) {
    dev_ctx.template Alloc<RealT>(out);
    return;
  }

  const DenseTensor real = Real<T, XPUContext>(dev_ctx, x);
  const DenseTensor imag = Imag<T, XPUContext>(dev_ctx, x);
  const DenseTensor real_sq = Multiply<RealT, XPUContext>(dev_ctx, real, real);
  const DenseTensor imag_sq = Multiply<RealT, XPUContext>(dev_ctx, imag, imag);
  const DenseTensor sum_sq = Add<RealT, XPUContext>(dev_ctx, real_sq, imag_sq);
  SqrtKernel<RealT, XPUContext>(dev_ctx, sum_sq, out);
}
#endif
}  // namespace phi

PD_REGISTER_KERNEL(abs,
                   XPU,
                   ALL_LAYOUT,
                   phi::AbsKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int8_t,
                   int32_t,
                   int64_t
#ifdef PADDLE_WITH_XPU_FFT
                   ,
                   phi::complex64
#endif
) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
