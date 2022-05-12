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

#include "paddle/phi/kernels/selected_rows/scale_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/scale_kernel.h"
namespace phi {
namespace sr {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const SelectedRows& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 SelectedRows* out) {
  if (x.value().Holder() != out->value().Holder() ||
      x.value().data() != out->value().data()) {
    out->set_rows(x.rows());
    out->set_height(x.height());
  }
  phi::ScaleKernel<T>(
      dev_ctx, x.value(), scale, bias, bias_after_scale, out->mutable_value());
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(scale_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::ScaleKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(scale_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::ScaleKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
#endif
