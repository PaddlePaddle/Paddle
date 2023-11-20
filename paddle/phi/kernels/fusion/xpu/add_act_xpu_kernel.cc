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
void AddActXPUKernel(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& x_max,
                     const DenseTensor& y,
                     const paddle::optional<DenseTensor>& y_max,
                     int act_type,
                     DenseTensor* out,
                     DenseTensor* out_max) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  const float* x_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto* y_data = reinterpret_cast<const XPUType*>(y.data<T>());
  const float* y_max_data =
      y_max.get_ptr() == nullptr ? nullptr : y_max.get_ptr()->data<float>();
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));

  std::vector<int64_t> x_shape = common::vectorize(x.dims());
  std::vector<int64_t> y_shape = common::vectorize(y.dims());
  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  int r =
      xpu::add_activation_fusion<XPUType, XPUType, XPUType>(  // TX/TY/TZ/TID
          /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
          /* const TX* x */ x_data,
          /* const TY* y */ y_data,
          /* TZ* z */ out_data,
          /* const std::vector<int64_t>& x_shape */ x_shape,
          /* const std::vector<int64_t>& y_shape */ y_shape,
          /* const float* max_x */ x_max_data,
          /* const float* max_y */ y_max_data,
          /* float* max_z */ ctx.template Alloc<float>(out_max),
          /* const baidu::xpu::api::Activation_t& act */ act);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add_act_xpu");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(add_act_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::AddActXPUKernel,
                   float,
                   phi::dtype::float16) {}
