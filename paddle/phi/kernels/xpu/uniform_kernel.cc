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

#include "paddle/phi/kernels/uniform_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void UniformKernel(const Context &dev_ctx,
                   const IntArray &shape,
                   DataType dtype,
                   const Scalar &min,
                   const Scalar &max,
                   int seed,
                   DenseTensor *out) {
  out->Resize(common::make_ddim(shape.GetData()));
  T *data = dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  int64_t real_seed = seed != 0 ? seed : dev_ctx.GetGenerator()->Random64();

  // algo:
  //       0: philox4x32_10_pytorch
  //       1: mt
  //       2: philox4x32_10_curand
  int algo = 0;
  int r = xpu::uniform<XPUType>(dev_ctx.x_context(),
                                reinterpret_cast<XPUType *>(data),
                                out->numel(),
                                static_cast<XPUType>(min.to<float>()),
                                static_cast<XPUType>(max.to<float>()),
                                real_seed,
                                algo);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "uniform");
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform,
                   XPU,
                   ALL_LAYOUT,
                   phi::UniformKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
