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

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
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
  auto max_ = max.to<T>();
  auto min_ = min.to<T>();

  PADDLE_ENFORCE_LE(
      min_,
      max_,
      errors::InvalidArgument("max should be greater than or equal to min. "
                              "But received min = %f, max = %f",
                              static_cast<float>(min_),
                              static_cast<float>(max_)));

  dev_ctx.template Alloc<T>(out);
  if (out && out->numel() == 0) return;
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  auto x_data = reinterpret_cast<const XPUDataType*>(x.data<T>());
  auto out_data = reinterpret_cast<XPUDataType*>(out->data<T>());
  int r = xpu::clamp(dev_ctx.x_context(),
                     x_data,
                     out_data,
                     x.numel(),
                     static_cast<XPUDataType>(min_),
                     static_cast<XPUDataType>(max_));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "clamp");
}

}  // namespace phi

PD_REGISTER_KERNEL(clip,
                   XPU,
                   ALL_LAYOUT,
                   phi::ClipKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int) {}
