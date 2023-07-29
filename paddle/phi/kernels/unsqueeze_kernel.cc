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

#include "paddle/phi/kernels/unsqueeze_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"

namespace phi {
template <typename T, typename Context>
void UnsqueezeInferKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const IntArray& axes,
                          DenseTensor* out) {
  auto x_dims = x.dims();
  auto out_dims = out->dims();
  if (axes.FromTensor()) {
    out_dims = funcs::GetUnsqueezeShape(axes.GetData(), x_dims);
  }
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  if (x.Holder() == out->Holder()) {
    return;
  }
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  out->Resize(out_dims);  // copy will reset the dims.
}

template <typename T, typename Context>
void UnsqueezeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& axes,
                     DenseTensor* out,
                     DenseTensor* xshape UNUSED) {
  UnsqueezeInferKernel<T, Context>(dev_ctx, x, axes, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(unsqueeze_infer,
                   CPU,
                   ALL_LAYOUT,
                   phi::UnsqueezeInferKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   int16_t,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(unsqueeze,
                   CPU,
                   ALL_LAYOUT,
                   phi::UnsqueezeKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   int16_t,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || defined(PADDLE_WITH_MUSAAA)
PD_REGISTER_KERNEL(unsqueeze_infer,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnsqueezeInferKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   int16_t,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(unsqueeze,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnsqueezeKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   int16_t,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(unsqueeze_infer,
                   XPU,
                   ALL_LAYOUT,
                   phi::UnsqueezeInferKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t) {}

PD_REGISTER_KERNEL(unsqueeze,
                   XPU,
                   ALL_LAYOUT,
                   phi::UnsqueezeKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t) {}
#endif
