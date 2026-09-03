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

#include "paddle/phi/kernels/frame_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "xpu/refactor/paddle_api.h"

namespace phi {

template <typename T, typename Context>
void FrameKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int frame_length,
                 int hop_length,
                 int axis,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0 || out->numel() == 0) {
    return;
  }

  auto xshape = common::vectorize<int64_t>(x.dims());
  auto outshape = common::vectorize<int64_t>(out->dims());
  int r = baidu::xpu::api::frame<XPUType>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x.data<T>()),
      reinterpret_cast<XPUType*>(out->data<T>()),
      xshape,
      outshape,
      static_cast<int64_t>(frame_length),
      static_cast<int64_t>(hop_length),
      static_cast<int64_t>(axis));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "frame");
}

}  // namespace phi

PD_REGISTER_KERNEL(frame,
                   XPU,
                   ALL_LAYOUT,
                   phi::FrameKernel,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
