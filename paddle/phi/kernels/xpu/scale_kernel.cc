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

#include "paddle/phi/kernels/scale_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      x.dims(),
      out->dims(),
      phi::errors::InvalidArgument("In and out should have the same dim,"
                                   " expected %s, but got %s.",
                                   x.dims().to_str().c_str(),
                                   out->dims().to_str().c_str()));
  using XPUType = typename XPUTypeTrait<T>::Type;
  int r = xpu::scale(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(x.data<T>()),
                     reinterpret_cast<XPUType*>(out->data<T>()),
                     x.numel(),
                     bias_after_scale,
                     scale.to<float>(),
                     bias);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
}

}  // namespace phi

PD_REGISTER_KERNEL(scale,
                   XPU,
                   ALL_LAYOUT,
                   phi::ScaleKernel,
                   float,
                   phi::dtype::float16,
                   int64_t) {}
