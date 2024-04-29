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

#include "paddle/phi/kernels/group_norm_kernel.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <string>

#include "paddle/common/layout.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout_str,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  const auto x_dims = common::vectorize<int>(x.dims());
  const int N = x_dims[0];
  const bool channel_first =
      data_layout == DataLayout::kNCHW || data_layout == DataLayout::kNCDHW;
  const int C = (channel_first ? x_dims[1] : x_dims[x_dims.size() - 1]);
  const int L =
      (channel_first
           ? std::accumulate(
                 x_dims.begin() + 2, x_dims.end(), 1, std::multiplies<int>())
           : std::accumulate(x_dims.begin() + 1,
                             x_dims.end() - 1,
                             1,
                             std::multiplies<int>()));

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<T>(mean);
  dev_ctx.template Alloc<T>(var);

  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();
  auto* mean_data = mean->data<T>();
  auto* var_data = var->data<T>();

  const T* scale_data = nullptr;
  if (scale_ptr) scale_data = scale_ptr->data<T>();
  const T* bias_data = nullptr;
  if (bias_ptr) bias_data = bias_ptr->data<T>();

  auto r =
      xpu::group_norm<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x_data),
                               reinterpret_cast<XPUType*>(y_data),
                               N,
                               C,
                               L,
                               1,
                               groups,
                               static_cast<XPUType>(epsilon),
                               reinterpret_cast<const XPUType*>(scale_data),
                               reinterpret_cast<const XPUType*>(bias_data),
                               reinterpret_cast<XPUType*>(mean_data),
                               reinterpret_cast<XPUType*>(var_data),
                               channel_first);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "group_norm");
}

}  // namespace phi

PD_REGISTER_KERNEL(group_norm, XPU, ALL_LAYOUT, phi::GroupNormKernel, float) {}
