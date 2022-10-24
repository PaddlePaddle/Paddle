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

#include "paddle/phi/kernels/squeeze_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"

namespace phi {
template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& axes,
                   DenseTensor* out) {
  auto x_dims = x.dims();
  std::vector<int32_t> tmp(axes.GetData().begin(), axes.GetData().end());
  auto out_dims = funcs::GetOutputSqueezeShape(tmp, x_dims, true);
  out->Resize(out_dims);

  dev_ctx.template Alloc<T>(out);
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  out->Resize(out_dims);  // copy will reset the dims.
}

template <typename T, typename Context>
void SqueezeWithXShapeKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const IntArray& axes,
                             DenseTensor* out,
                             DenseTensor* xshape) {
  SqueezeKernel<T, Context>(dev_ctx, x, axes, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(squeeze,
                   CPU,
                   ALL_LAYOUT,
                   phi::SqueezeKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(squeeze_with_xshape,
                   CPU,
                   ALL_LAYOUT,
                   phi::SqueezeWithXShapeKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(squeeze,
                   GPU,
                   ALL_LAYOUT,
                   phi::SqueezeKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(squeeze_with_xshape,
                   GPU,
                   ALL_LAYOUT,
                   phi::SqueezeWithXShapeKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(squeeze,
                   XPU,
                   ALL_LAYOUT,
                   phi::SqueezeKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t) {}

PD_REGISTER_KERNEL(squeeze_with_xshape,
                   XPU,
                   ALL_LAYOUT,
                   phi::SqueezeWithXShapeKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t) {}
#endif
