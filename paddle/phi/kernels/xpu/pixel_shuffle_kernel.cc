// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/pixel_shuffle_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void PixelShuffleKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int upscale_factor,
                        const std::string& data_format,
                        DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const T* x_ptr = x.data<T>();
  T* y_ptr = dev_ctx.template Alloc<T>(out);
  if (out && out->numel() == 0) {
    return;
  }

  bool is_nchw = data_format == "NCHW";

  int64_t n = x.dims()[0];
  int64_t xc = x.dims()[is_nchw ? 1 : 3];
  int64_t xh = x.dims()[is_nchw ? 2 : 1];
  int64_t xw = x.dims()[is_nchw ? 3 : 2];

  int r = pixel_shuffle(dev_ctx.x_context(),
                        reinterpret_cast<const XPUType*>(x_ptr),
                        reinterpret_cast<XPUType*>(y_ptr),
                        n,
                        xc,
                        xh,
                        xw,
                        upscale_factor,
                        is_nchw);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pixel_shuffle");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    pixel_shuffle, XPU, ALL_LAYOUT, phi::PixelShuffleKernel, float) {}
