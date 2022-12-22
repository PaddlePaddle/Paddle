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

#include "paddle/phi/kernels/activation_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     DenseTensor* out) {
  HardSwishRawKernel<T, Context>(dev_ctx, x, 6, 6, 3, out);
}

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  Relu6RawKernel<T, Context>(dev_ctx, x, 6, out);
}

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  SwishRawKernel<T, Context>(dev_ctx, x, 1.0, out);
}

}  // namespace phi
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(
    hard_swish, CPU, ALL_LAYOUT, phi::HardSwishKernel, float, double) {}
PD_REGISTER_KERNEL(relu6, CPU, ALL_LAYOUT, phi::Relu6Kernel, float, double) {}
PD_REGISTER_KERNEL(swish, CPU, ALL_LAYOUT, phi::SwishKernel, float, double) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(hard_swish,
                   GPU,
                   ALL_LAYOUT,
                   phi::HardSwishKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(relu6,
                   GPU,
                   ALL_LAYOUT,
                   phi::Relu6Kernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(swish,
                   GPU,
                   ALL_LAYOUT,
                   phi::SwishKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#endif

#if defined PADDLE_WITH_XPU
PD_REGISTER_KERNEL(hard_swish, XPU, ALL_LAYOUT, phi::HardSwishKernel, float) {}
PD_REGISTER_KERNEL(relu6, XPU, ALL_LAYOUT, phi::Relu6Kernel, float) {}
PD_REGISTER_KERNEL(swish, XPU, ALL_LAYOUT, phi::SwishKernel, float) {}
#endif

#ifdef PADDLE_WITH_MKLDNN
PD_REGISTER_KERNEL(hard_swish,
                   OneDNN,
                   ONEDNN,
                   phi::HardSwishKernel,
                   float,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(
    relu6, OneDNN, ONEDNN, phi::Relu6Kernel, float, phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(
    swish, OneDNN, ONEDNN, phi::SwishKernel, float, phi::dtype::bfloat16) {}
#endif
