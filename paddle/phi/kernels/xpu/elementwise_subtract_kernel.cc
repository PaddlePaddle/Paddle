/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/kernels/xpu/elementwise.h"
namespace phi {

template <typename T, typename Context>
void SubtractRawKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       int axis,
                       DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  phi::XPUElementwise<T, XPUType>(
  dev_ctx, x, y, axis, out, xpu::broadcast_sub<XPUType>);
}

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out){
	SubtractRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

}  // namespace phi
PD_REGISTER_KERNEL(subtract_raw, XPU, ALL_LAYOUT,phi::SubtractRawKernel,float,phi::dtype::float16) {}

PD_REGISTER_KERNEL(substract, XPU, ALL_LAYOUT, phi::SubtractKernel,float, phi::dtype::float16) {}
