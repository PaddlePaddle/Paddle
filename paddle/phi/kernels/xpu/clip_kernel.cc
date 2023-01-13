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

#include "paddle/phi/kernels/clip_kernel.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ClipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& min,
                const Scalar& max,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  auto x_data = reinterpret_cast<const XPUDataType*>(x.data<T>());
  auto out_data = reinterpret_cast<XPUDataType*>(out->data<T>());
  int r = xpu::clip_v2(dev_ctx.x_context(),
                       x_data,
                       out_data,
                       x.numel(),
                       min.to<float>(),
                       max.to<float>());

  PADDLE_ENFORCE_EQ(r,
                    XPU_SUCCESS,
                    phi::errors::External("XPU API(clip_v2) return wrong "
                                          "value[%d %s]",
                                          r,
                                          XPUAPIErrorMsg[r]));
}

}  // namespace phi

PD_REGISTER_KERNEL(clip, XPU, ALL_LAYOUT, phi::ClipKernel, float) {}
