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

#include "paddle/pten/kernels/scale_kernel.h"

#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/pten/backends/xpu/xpu_context.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/float16.h"
#include "paddle/pten/core/compat/convert_utils.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  out->mutable_data<T>(dev_ctx.GetPlace());

  PADDLE_ENFORCE_EQ(x.dims(),
                    out->dims(),
                    paddle::platform::errors::InvalidArgument(
                        "In and out should have the same dim,"
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
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      paddle::platform::errors::External(
          "XPU scale kernel return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
}

}  // namespace pten

PT_REGISTER_KERNEL(scale,
                   XPU,
                   ALL_LAYOUT,
                   pten::ScaleKernel,
                   float,
                   pten::dtype::float16,
                   int64_t) {}
