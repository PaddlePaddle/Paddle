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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void SinePosXPUKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  // fix precision of fp16 model
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  std::vector<int64_t> x_shape = phi::vectorize(x.dims());
  std::vector<int64_t> y_shape = phi::vectorize(y.dims());
  // yolo_box_coord only support fp32&&fp16 precision
  int r = xpu::sine_pos_fusion<XPUType>(
      /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
      /* const T* x */ x_data,
      /* const T* y */ y_data,
      /* T* out */ out_data,
      /* int64_t batch */ x_shape[0],
      /* int64_t n */ x_shape[1],
      /* int64_t dim */ y_shape[0]);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sine_pos_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(sine_pos_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::SinePosXPUKernel,
                   float,
                   phi::dtype::float16) {}
